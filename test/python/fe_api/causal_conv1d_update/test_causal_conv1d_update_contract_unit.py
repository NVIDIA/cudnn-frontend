# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-contract tests that do not compile or launch a GPU kernel."""

import importlib
import inspect
import os
import re
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
        from cudnn.causal_conv1d_update_sm100 import _CausalConv1dUpdatePlan
    except ImportError as exc:
        pytest.skip(f"CuTe DSL dependencies unavailable: {exc}")
    return _CausalConv1dUpdatePlan


def _inputs(*, n_rows=2, n_channels=8, n_slots=3, state_len=4, indexed=True):
    x = torch.zeros(n_rows, n_channels, dtype=torch.bfloat16)
    weight = torch.zeros(n_channels, 4, dtype=torch.bfloat16)
    state = torch.zeros(n_slots, n_channels, state_len, dtype=torch.bfloat16)
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
    api_module = sys.modules[api.__class__.__module__]
    monkeypatch.setattr(api_module, "_as_torch_stream", lambda current_stream, device: object())
    monkeypatch.setattr(api_module, "_record_streams", lambda tensors, stream: None)


def test_only_cudnn_ops_exports_the_semantic_api(tmp_path):
    # Test a fresh interpreter against this checkout's __init__.py rather than
    # the prebuilt package which conftest overlays for other contract tests.
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
from cudnn.ops import causal_conv1d_update
import cudnn.causal_conv1d_update_sm100 as implementation

assert callable(causal_conv1d_update) and not isinstance(causal_conv1d_update, types.ModuleType)
assert cudnn.ops.causal_conv1d_update is causal_conv1d_update
assert not hasattr(cudnn, "causal_conv1d_update")
assert not hasattr(cudnn, "causal_conv1d_update_wrapper_sm100")
assert not hasattr(cudnn, "CausalConv1dUpdateSm100")
assert implementation.__all__ == []
assert not hasattr(implementation, "causal_conv1d_update")
assert not hasattr(implementation, "causal_conv1d_update_wrapper_sm100")
assert not hasattr(implementation, "CausalConv1dUpdateSm100")
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


def test_public_semantic_signature():
    from cudnn.ops import causal_conv1d_update

    parameters = inspect.signature(causal_conv1d_update).parameters
    assert tuple(parameters) == (
        "x",
        "conv_state",
        "weight",
        "bias",
        "activation",
        "cache_seqlens",
        "conv_state_indices",
    )
    assert parameters["bias"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["bias"].default is None
    assert parameters["activation"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["activation"].default is None
    assert parameters["cache_seqlens"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["cache_seqlens"].default is None
    assert parameters["conv_state_indices"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["conv_state_indices"].default is None


def test_public_custom_op_declares_state_mutation():
    from cudnn.ops import causal_conv1d_update  # noqa: F401

    schema = str(torch.ops.cudnn._causal_conv1d_update.default._schema)
    assert re.search(r"Tensor\(a\d*!\) conv_state", schema)
    assert schema.count("!") == 1
    assert "Tensor? cache_seqlens" in schema
    assert "Tensor? conv_state_indices" in schema
    assert schema.endswith(" -> Tensor")


def test_private_raw_op_canonicalizes_activation_before_native_call(monkeypatch):
    ops_module = importlib.import_module("cudnn.ops._causal_conv1d_update")
    implementation = importlib.import_module("cudnn.causal_conv1d_update_sm100")
    x, weight, state, _, indices = _inputs()
    calls = []
    sentinel = object()

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(implementation, "_causal_conv1d_update", native)

    result = ops_module._causal_conv1d_update_primitive._init_fn(x, state, weight, None, "swish", None, indices)

    assert result is sentinel
    assert calls == [((x, state, weight, None, "silu"), {"conv_state_indices": indices})]


def test_semantic_shape_contract_is_not_hard_coded_to_width_four():
    ops_module = importlib.import_module("cudnn.ops._causal_conv1d_update")
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    bias = torch.zeros(8, dtype=torch.bfloat16)
    cache_seqlens = torch.zeros(2, dtype=torch.int32)
    indices = torch.tensor([4, 1], dtype=torch.int32)

    # Minimum legal state length (L = W - 1).
    ops_module._validate_semantic_contract(
        x,
        torch.zeros(5, 8, 2, dtype=torch.bfloat16),
        torch.zeros(8, 3, dtype=torch.bfloat16),
        bias,
        cache_seqlens,
        indices,
    )
    # Longer state remains part of the same semantic contract.
    ops_module._validate_semantic_contract(
        x,
        torch.zeros(5, 8, 7, dtype=torch.bfloat16),
        torch.zeros(8, 4, dtype=torch.bfloat16),
        bias,
        None,
        indices,
    )


def test_semantic_shape_contract_rejects_state_shorter_than_history():
    ops_module = importlib.import_module("cudnn.ops._causal_conv1d_update")
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    state = torch.zeros(2, 8, 2, dtype=torch.bfloat16)
    weight = torch.zeros(8, 4, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=r"L >= W - 1, got L=2, W=4"):
        ops_module._validate_semantic_contract(x, state, weight, None, None, None)


@pytest.mark.parametrize(
    "x_shape,state_shape,match",
    [
        ((0, 8), (1, 8, 4), "row count N must be positive"),
        ((2, 0), (2, 0, 4), "channel count D must be positive"),
        ((2, 8), (0, 8, 4), "slot count S must be positive"),
    ],
)
def test_semantic_shape_contract_rejects_empty_native_extents(x_shape, state_shape, match):
    ops_module = importlib.import_module("cudnn.ops._causal_conv1d_update")
    x = torch.zeros(x_shape, dtype=torch.bfloat16)
    state = torch.zeros(state_shape, dtype=torch.bfloat16)
    weight = torch.zeros(x_shape[1], 4, dtype=torch.bfloat16)
    indices = torch.zeros(x_shape[0], dtype=torch.int32) if state_shape[0] == 0 else None

    with pytest.raises(ValueError, match=match):
        ops_module._validate_semantic_contract(x, state, weight, None, None, indices)


def test_public_meta_tensor_uses_registered_fake_kernel():
    from cudnn.ops import causal_conv1d_update

    x = torch.empty(2, 8, device="meta", dtype=torch.bfloat16)
    state = torch.empty(2, 8, 4, device="meta", dtype=torch.bfloat16)
    weight = torch.empty(8, 4, device="meta", dtype=torch.bfloat16)

    output = causal_conv1d_update(x, state, weight)

    assert output.device.type == "meta"
    assert output.shape == x.shape
    assert output.stride() == x.stride()


@pytest.mark.parametrize(
    "state_len,width,has_cache",
    [(2, 3, False), (5, 4, False), (4, 4, True)],
    ids=["width-three", "state-length-five", "circular-buffer"],
)
def test_semantically_valid_unimplemented_configs_decline_clearly(state_len, width, has_cache):
    from cudnn.ops import causal_conv1d_update
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode(), pytest.raises(NotImplementedError, match="current native.*supports only"):
        x = torch.empty(2, 8, device="cuda", dtype=torch.bfloat16)
        state = torch.empty(2, 8, state_len, device="cuda", dtype=torch.bfloat16)
        weight = torch.empty(8, width, device="cuda", dtype=torch.bfloat16)
        cache_seqlens = torch.empty(2, device="cuda", dtype=torch.int32) if has_cache else None
        causal_conv1d_update(x, state, weight, cache_seqlens=cache_seqlens)


def test_multi_token_update_is_reserved_but_not_silently_interpreted():
    ops_module = importlib.import_module("cudnn.ops._causal_conv1d_update")
    x = torch.zeros(2, 8, 3, dtype=torch.bfloat16)
    state = torch.zeros(2, 8, 4, dtype=torch.bfloat16)
    weight = torch.zeros(8, 4, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=r"x must have shape \[N, D\]"):
        ops_module._validate_semantic_contract(x, state, weight, None, None, None)


@pytest.mark.parametrize("state_len", [3, 4], ids=["w-minus-one", "legacy-four"])
def test_public_fake_tensor_contract_preserves_output_metadata(state_len):
    from cudnn.ops import causal_conv1d_update
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(2, 8, device="cuda", dtype=torch.bfloat16)
        state = torch.empty(3, 8, state_len, device="cuda", dtype=torch.bfloat16)
        weight = torch.empty(8, 4, device="cuda", dtype=torch.bfloat16)
        indices = torch.empty(2, device="cuda", dtype=torch.int32)
        output = causal_conv1d_update(x, state, weight, activation="swish", conv_state_indices=indices)

    assert isinstance(output, FakeTensor)
    assert output.shape == x.shape
    assert output.stride() == x.stride()
    assert output.dtype == x.dtype
    assert output.device == x.device


def test_activation_aliases_and_cache_keys_are_canonical():
    ops_module = importlib.import_module("cudnn.ops._causal_conv1d_update")
    api_module = importlib.import_module("cudnn.causal_conv1d_update_sm100.api")
    x, weight, state, _, indices = _inputs()

    assert ops_module._normalize_activation(None) == "identity"
    assert ops_module._normalize_activation("identity") == "identity"
    assert ops_module._normalize_activation("silu") == "silu"
    assert ops_module._normalize_activation("swish") == "silu"
    with pytest.raises(ValueError, match="activation must be"):
        ops_module._normalize_activation("relu")

    identity_key = api_module._cache_key(x, state, weight, indices, None, "identity")
    silu_key = api_module._cache_key(x, state, weight, indices, None, "silu")
    assert identity_key != silu_key


def test_valid_descriptor_contract_without_kernel(monkeypatch):
    cls = _api_class()
    x, weight, state, output, indices = _inputs()
    api = cls(x, weight, state, output, indices)
    _mock_cuda_contract(monkeypatch, api)

    assert api.check_support()
    assert (api.n_rows, api.n_channels, api.n_slots) == (2, 8, 3)


def test_width_four_minimum_state_descriptor_contract_without_kernel(monkeypatch):
    cls = _api_class()
    x, weight, state, output, indices = _inputs(state_len=3)
    api = cls(x, weight, state, output, indices)
    _mock_cuda_contract(monkeypatch, api)

    assert api.check_support()
    assert (api.n_rows, api.n_channels, api.n_slots, api.state_len) == (2, 8, 3, 3)


def test_optional_bias_descriptor_contract_without_kernel(monkeypatch):
    cls = _api_class()
    x, weight, state, output, indices = _inputs()
    bias = torch.zeros(x.shape[1], dtype=torch.bfloat16)
    api = cls(x, weight, state, output, indices, bias)
    _mock_cuda_contract(monkeypatch, api)

    assert api.check_support()
    assert api.bias_desc.shape == (x.shape[1],)

    bad_bias = torch.zeros(x.shape[1] + 1, dtype=torch.bfloat16)
    bad_api = cls(x, weight, state, output, indices, bad_bias)
    _mock_cuda_contract(monkeypatch, bad_api)
    with pytest.raises(ValueError, match="Bias tensor shape mismatch"):
        bad_api.check_support()


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
def test_supported_architecture_gate(monkeypatch, capability):
    cls = _api_class()
    api = cls(*_inputs())
    _mock_cuda_contract(monkeypatch, api, capability=capability)

    assert api.check_support()


@pytest.mark.parametrize("capability", [(7, 5), (7, 9), (11, 1)])
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

    bias = torch.zeros(x.shape[1], dtype=torch.bfloat16)
    biased_api = cls(x, weight, state, output, indices, bias)
    _mock_cuda_contract(monkeypatch, biased_api)
    assert biased_api.check_support()
    biased_api._compiled_kernel = lambda *args: None
    with pytest.raises(ValueError, match="bias presence must match"):
        biased_api.execute(
            x,
            weight,
            state,
            output,
            indices,
            current_stream=stream,
        )
    biased_api.execute(
        x,
        weight,
        state,
        output,
        indices,
        current_stream=stream,
        bias_tensor=bias,
    )

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


def test_overlap_uses_exact_contiguous_byte_spans():
    api_module = importlib.import_module("cudnn.causal_conv1d_update_sm100.api")

    class Span:
        def __init__(self, address, numel, element_size):
            self.address = address
            self.length = numel
            self.itemsize = element_size

        def data_ptr(self):
            return self.address

        def numel(self):
            return self.length

        def element_size(self):
            return self.itemsize

    left = Span(0x1000, 4, 2)

    assert api_module._tensors_overlap(left, Span(0x1006, 2, 2))
    assert api_module._tensors_overlap(Span(0x0FFE, 2, 2), left)
    assert not api_module._tensors_overlap(left, Span(0x1008, 2, 2))


def test_record_streams_covers_every_present_raw_pointer_operand():
    api_module = importlib.import_module("cudnn.causal_conv1d_update_sm100.api")
    consumer = object()

    class TensorRecorder:
        def __init__(self):
            self.streams = []

        def record_stream(self, stream):
            self.streams.append(stream)

    first = TensorRecorder()
    second = TensorRecorder()
    api_module._record_streams((first, None, second), consumer)
    api_module._record_streams((first, second), None)

    assert first.streams == [consumer]
    assert second.streams == [consumer]


def test_execute_records_operands_on_resolved_explicit_stream(monkeypatch):
    api_module = importlib.import_module("cudnn.causal_conv1d_update_sm100.api")
    cls = _api_class()
    x, weight, state, output, indices = _inputs()
    bias = torch.zeros(x.shape[1], dtype=torch.bfloat16)
    api = cls(x, weight, state, output, indices, bias)
    _mock_cuda_contract(monkeypatch, api)
    assert api.check_support()

    launches = []
    recorded = []
    resolved = []
    consumer = object()
    stream = cuda.CUstream(17)
    api._compiled_kernel = lambda *args: launches.append(args)

    def resolve(current_stream, device):
        resolved.append((current_stream, device))
        return consumer

    monkeypatch.setattr(api_module, "_as_torch_stream", resolve)
    monkeypatch.setattr(api_module, "_record_streams", lambda tensors, stream: recorded.append((tensors, stream)))

    result = api.execute(
        x,
        weight,
        state,
        output,
        indices,
        current_stream=stream,
        bias_tensor=bias,
    )

    assert result is output
    assert resolved == [(stream, x.device)]
    assert len(launches) == 1
    assert launches[0][-1] == stream
    assert recorded == [((x, weight, bias, state, output, indices), consumer)]


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
