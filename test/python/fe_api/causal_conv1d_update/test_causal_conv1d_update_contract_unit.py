# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-contract tests that do not compile or launch a GPU kernel."""

import inspect
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from cuda.bindings import driver as cuda

pytestmark = pytest.mark.L0

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


def _api_class():
    try:
        from cudnn.causal_conv1d_update_sm100 import CausalConv1dUpdateSm100
    except ImportError as exc:
        pytest.skip(f"CuTe DSL dependencies unavailable: {exc}")
    return CausalConv1dUpdateSm100


def _inputs(*, n_rows=2, n_channels=8, n_slots=3, indexed=True):
    x = torch.zeros(n_rows, n_channels, dtype=torch.bfloat16)
    weight = torch.zeros(n_channels, 4, dtype=torch.bfloat16)
    state = torch.zeros(n_slots, n_channels, 4, dtype=torch.bfloat16)
    output = torch.empty_like(x)
    indices = torch.arange(n_rows, dtype=torch.int32) if indexed else None
    return x, weight, state, output, indices


def _metadata_desc(tensor):
    if tensor is None:
        return None
    from cudnn.api_base import TensorDesc

    shape = tuple(tensor.shape)
    stride = tuple(tensor.stride())
    return TensorDesc(
        dtype=tensor.dtype,
        shape=shape,
        stride=stride,
        stride_order=TensorDesc._compute_stride_order(shape, stride),
        device=torch.device("cuda"),
    )


def _mock_cuda_contract(monkeypatch, api, capability=(10, 0)):
    # These tests exercise descriptor/contract logic on CPU tensors only.  The
    # real GPU suite independently checks the CUDA-device and architecture gate.
    monkeypatch.setattr(api, "_require_cuda", lambda desc, name: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)


def test_public_exports_remain_callables_not_modules(tmp_path):
    # A same-named ``cudnn.causal_conv1d_update`` package previously replaced
    # the lazy top-level function with the imported module object. Test a fresh
    # interpreter against this checkout's __init__.py rather than the prebuilt
    # package which conftest overlays for the other host-contract tests.
    import cudnn

    source = Path(__file__).resolve().parents[4] / "python" / "cudnn"
    probe = tmp_path / "cudnn"
    shutil.copytree(source, probe)
    compiled_modules = list(Path(cudnn.__file__).resolve().parent.glob("_compiled_module*.so"))
    assert len(compiled_modules) == 1
    (probe / compiled_modules[0].name).symlink_to(compiled_modules[0])

    script = """
import types
import cudnn
import cudnn.ops
from cudnn.causal_conv1d_update_sm100 import (
    CausalConv1dUpdateSm100,
    causal_conv1d_update,
    causal_conv1d_update_wrapper_sm100,
)

exports = (
    cudnn.causal_conv1d_update,
    cudnn.causal_conv1d_update_wrapper_sm100,
    cudnn.CausalConv1dUpdateSm100,
    cudnn.ops.causal_conv1d_update,
)
assert all(callable(export) and not isinstance(export, types.ModuleType) for export in exports)
assert cudnn.causal_conv1d_update is causal_conv1d_update
assert cudnn.causal_conv1d_update_wrapper_sm100 is causal_conv1d_update_wrapper_sm100
assert cudnn.CausalConv1dUpdateSm100 is CausalConv1dUpdateSm100
assert cudnn.ops.causal_conv1d_update is causal_conv1d_update
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


def test_public_helpers_use_state_before_weight_positional_order():
    from cudnn.causal_conv1d_update_sm100 import (
        causal_conv1d_update,
        causal_conv1d_update_wrapper_sm100,
    )

    expected = ("x", "state", "weight", "state_indices")
    assert tuple(inspect.signature(causal_conv1d_update).parameters)[:4] == expected
    assert tuple(inspect.signature(causal_conv1d_update_wrapper_sm100).parameters)[:4] == expected


def test_valid_descriptor_contract_without_kernel(monkeypatch):
    cls = _api_class()
    x, weight, state, output, indices = _inputs()
    api = cls(x, weight, state, output, indices)
    _mock_cuda_contract(monkeypatch, api)

    assert api.check_support()
    assert (api.n_rows, api.n_channels, api.n_slots) == (2, 8, 3)


def test_metadata_only_descriptors_skip_sample_pointer_alignment(monkeypatch):
    cls = _api_class()
    api = cls(*(_metadata_desc(tensor) for tensor in _inputs()))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))

    assert api._sample_alignment_remainders == {}
    assert api.check_support()


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda x, w, s, o, i: (x, w[:, :3], s, o, i), "Weight tensor shape mismatch"),
        (
            lambda x, w, s, o, i: (
                torch.empty((x.shape[0], x.shape[1] * 2), dtype=x.dtype)[:, ::2],
                w,
                s,
                o,
                i,
            ),
            "X tensor stride mismatch",
        ),
        (lambda x, w, s, o, i: (x.float(), w, s, o.float(), i), "X dtype mismatch"),
        (lambda x, w, s, o, i: (x, w, s[:1], o, None), "State needs at least N slots"),
        (
            lambda x, w, s, o, i: (x, w, s, o, i.to(torch.int64)),
            "State indices dtype mismatch",
        ),
    ],
    ids=["kernel-width", "shape", "dtype", "too-few-slots", "index-dtype"],
)
def test_bad_descriptor_contract_fails_before_compile(monkeypatch, mutate, match):
    cls = _api_class()
    args = mutate(*_inputs())
    api = cls(*args)
    _mock_cuda_contract(monkeypatch, api)

    with pytest.raises((TypeError, ValueError), match=match):
        api.check_support()


@pytest.mark.parametrize("capability", _SUPPORTED_COMPUTE_CAPABILITIES)
def test_functional_architecture_allowlist(monkeypatch, capability):
    cls = _api_class()
    api = cls(*_inputs())
    _mock_cuda_contract(monkeypatch, api, capability=capability)

    assert api.check_support()


@pytest.mark.parametrize("capability", [(7, 5), (10, 1), (11, 1), (13, 0)])
def test_unsupported_architecture_gate(monkeypatch, capability):
    cls = _api_class()
    api = cls(*_inputs())
    _mock_cuda_contract(monkeypatch, api, capability=capability)

    with pytest.raises(RuntimeError, match="supports compute capabilities"):
        api.check_support()


def test_sample_pointer_alignment_is_part_of_compile_contract(monkeypatch):
    cls = _api_class()
    x, weight, state, output, indices = _inputs()
    storage = torch.empty(x.numel() + 1, dtype=x.dtype)
    misaligned_x = storage[1:].view_as(x)
    api = cls(misaligned_x, weight, state, output, indices)
    _mock_cuda_contract(monkeypatch, api)

    with pytest.raises(ValueError, match="X data pointer must be 16-byte aligned"):
        api.check_support()


def test_execute_revalidates_presence_and_aliases(monkeypatch):
    cls = _api_class()
    x, weight, state, output, indices = _inputs()
    api = cls(x, weight, state, output, indices)
    _mock_cuda_contract(monkeypatch, api)
    assert api.check_support()

    calls = []
    api._compiled_kernel = lambda *args: calls.append(args)
    stream = cuda.CUstream(7)
    api.execute(x, weight, state, output, indices, current_stream=stream)
    assert len(calls) == 1

    with pytest.raises(ValueError, match="presence must match"):
        api.execute(x, weight, state, output, None, current_stream=stream)

    storage = torch.empty(x.numel() + 1, dtype=x.dtype)
    misaligned_x = storage[1:].view_as(x)
    with pytest.raises(ValueError, match="X data pointer must be 16-byte aligned"):
        api.execute(misaligned_x, weight, state, output, indices, current_stream=stream)

    overlapping_output = state.view(-1)[: x.numel()].view_as(x)
    with pytest.raises(ValueError, match="State must not overlap Output"):
        api.execute(x, weight, state, overlapping_output, indices, current_stream=stream)

    overlapping_indices = state.view(torch.int32).view(-1)[: x.shape[0]]
    with pytest.raises(ValueError, match="State must not overlap State indices"):
        api.execute(x, weight, state, output, overlapping_indices, current_stream=stream)

    # Four output rows and a width-four filter have the same storage size.
    # Rejecting this alias is required because row CTAs would otherwise race
    # while overwriting weights that other rows still need to read.
    aliased_storage = torch.empty(weight.numel(), dtype=torch.bfloat16)
    aliased_weight = aliased_storage.view_as(weight)
    aliased_output = aliased_storage.view(4, x.shape[1])
    x4 = torch.zeros_like(aliased_output)
    state4 = torch.zeros(4, state.shape[1], state.shape[2], dtype=state.dtype)
    indices4 = torch.arange(4, dtype=torch.int32)
    api4 = cls(x4, aliased_weight, state4, aliased_output, indices4)
    _mock_cuda_contract(monkeypatch, api4)
    assert api4.check_support()
    api4._compiled_kernel = lambda *args: None
    with pytest.raises(ValueError, match="Output must not overlap Weight"):
        api4.execute(
            x4,
            aliased_weight,
            state4,
            aliased_output,
            indices4,
            current_stream=stream,
        )


def test_execute_rejects_autograd(monkeypatch):
    cls = _api_class()
    x, weight, state, output, indices = _inputs()
    weight.requires_grad_(True)
    api = cls(x, weight, state, output, indices)
    _mock_cuda_contract(monkeypatch, api)
    assert api.check_support()
    api._compiled_kernel = lambda *args: None

    with pytest.raises(RuntimeError, match="inference-only"):
        api.execute(x, weight, state, output, indices, current_stream=cuda.CUstream(7))


def test_execute_rejects_requires_grad_output(monkeypatch):
    cls = _api_class()
    x, weight, state, output, indices = _inputs()
    output.requires_grad_(True)
    api = cls(x, weight, state, output, indices)
    _mock_cuda_contract(monkeypatch, api)
    assert api.check_support()
    api._compiled_kernel = lambda *args: None

    with pytest.raises(RuntimeError, match="inference-only"):
        api.execute(x, weight, state, output, indices, current_stream=cuda.CUstream(7))
