# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU correctness and contract tests for the decode-update operation."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from cuda.bindings import driver as cuda
from cudnn._causal_conv1d_arch import (
    is_supported_causal_conv1d_update_compute_capability,
)


def _has_supported_gpu() -> bool:
    return torch.cuda.is_available() and is_supported_causal_conv1d_update_compute_capability(torch.cuda.get_device_capability())


pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(not _has_supported_gpu(), reason="requires a functionally supported GPU architecture"),
]


def _load_api():
    try:
        from cudnn.causal_conv1d_update_sm100 import (
            _CausalConv1dUpdatePlan,
        )
        from cudnn.ops import causal_conv1d_update
    except ImportError as exc:
        pytest.skip(f"CuTe DSL dependencies unavailable: {exc}")
    return _CausalConv1dUpdatePlan, causal_conv1d_update


def _reference_step(x, weight, state, state_indices=None, bias=None, activation=None):
    """Explicit FP32 output reference plus exact BF16 state transition."""

    n_rows = x.shape[0]
    slots = torch.arange(n_rows, device=x.device, dtype=torch.long) if state_indices is None else state_indices.long()
    valid = slots >= 0
    output = torch.zeros_like(x)
    expected_state = state.clone()
    if valid.any():
        valid_slots = slots[valid]
        selected = state.index_select(0, valid_slots)
        history = torch.cat((selected, x[valid].unsqueeze(-1)), dim=-1)
        updated = history[..., -state.shape[-1] :]
        window = history[..., -weight.shape[-1] :]
        accumulator = (window.float() * weight.float().unsqueeze(0)).sum(dim=-1)
        if bias is not None:
            accumulator = accumulator + bias.float()
        if activation in ("silu", "swish"):
            accumulator = F.silu(accumulator)
        output[valid] = accumulator.to(torch.bfloat16)
        expected_state.index_copy_(0, valid_slots, updated)
    return output, expected_state


def _assert_state_bits_equal(actual, expected):
    torch.testing.assert_close(actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)


def _padded_x(values, row_stride):
    storage = torch.empty(
        values.shape[0],
        row_stride,
        device=values.device,
        dtype=values.dtype,
    )
    result = storage[:, : values.shape[1]]
    result.copy_(values)
    return result


@torch.no_grad()
def test_nonzero_state_across_consecutive_steps():
    plan_cls, _ = _load_api()
    torch.manual_seed(0)
    n_rows, n_channels = 3, 521
    state = torch.randn(n_rows, n_channels, 4, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n_channels, 4, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(n_rows, n_channels, device="cuda", dtype=torch.bfloat16)

    api = plan_cls(
        sample_x=torch.empty_like(output),
        sample_weight=weight,
        sample_state=state,
        sample_output=output,
    )
    assert api.check_support()
    api.compile()

    expected_state = state.clone()
    for _ in range(3):
        x = torch.randn(n_rows, n_channels, device="cuda", dtype=torch.bfloat16)
        expected_output, expected_state = _reference_step(x, weight, expected_state)
        api.execute(x, weight, state, output)
        torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
        _assert_state_bits_equal(state, expected_state)


@torch.no_grad()
@pytest.mark.parametrize("state_len", [3, 4], ids=["w-minus-one", "legacy-four"])
def test_padded_x_rows_match_reference_and_preserve_compact_output(state_len):
    _, causal_conv1d_update = _load_api()
    torch.manual_seed(53 + state_len)
    n_rows, n_channels, row_stride = 4, 257, 264
    dense_x = torch.randn(n_rows, n_channels, device="cuda", dtype=torch.bfloat16)
    x = _padded_x(dense_x, row_stride)
    weight = torch.randn(n_channels, 4, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(n_rows, n_channels, state_len, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n_channels, device="cuda", dtype=torch.bfloat16)
    expected_output, expected_state = _reference_step(dense_x, weight, state, bias=bias, activation="silu")

    assert x.stride() == (row_stride, 1)
    output = causal_conv1d_update(x, state, weight, bias, activation="silu")

    assert output.is_contiguous()
    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(state, expected_state)


@torch.no_grad()
@pytest.mark.parametrize("n_channels", [2048, 4096])
def test_n128_no_index_representative_shape_correctness(n_channels):
    _, causal_conv1d_update = _load_api()
    torch.manual_seed(29 + n_channels)
    n_rows = 128
    weight = torch.randn(n_channels, 4, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(n_rows, n_channels, 4, device="cuda", dtype=torch.bfloat16)
    expected_state = state.clone()

    # Use the public cached route for two updates at representative decode
    # shapes. Every architecture uses the same one-row schedule.
    for _ in range(2):
        x = torch.randn(n_rows, n_channels, device="cuda", dtype=torch.bfloat16)
        expected_output, expected_state = _reference_step(x, weight, expected_state)
        output = causal_conv1d_update(x, state, weight)
        torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
        _assert_state_bits_equal(state, expected_state)


@torch.no_grad()
@pytest.mark.parametrize("n_channels", [2048, 4096])
def test_n128_representative_shape_state_shift_is_bitwise(n_channels):
    _, causal_conv1d_update = _load_api()
    torch.manual_seed(41 + n_channels)
    n_rows = 128

    # Exercise arbitrary BF16 payloads through the public/cache route.  Ignore
    # output: NaN payloads make only the state mutation contract meaningful,
    # and that contract must remain bitwise over repeated steps.
    state_bits = torch.randint(
        -(2**15),
        2**15,
        (n_rows, n_channels, 4),
        device="cuda",
        dtype=torch.int16,
    )
    state = state_bits.view(torch.bfloat16)
    weight = torch.zeros(n_channels, 4, device="cuda", dtype=torch.bfloat16)
    expected_bits = state_bits.clone()

    for _ in range(2):
        x_bits = torch.randint(
            -(2**15),
            2**15,
            (n_rows, n_channels),
            device="cuda",
            dtype=torch.int16,
        )
        causal_conv1d_update(x_bits.view(torch.bfloat16), state, weight)
        expected_bits = torch.cat((expected_bits[..., 1:], x_bits.unsqueeze(-1)), dim=-1)
        torch.testing.assert_close(state.view(torch.int16), expected_bits, rtol=0, atol=0)


@torch.no_grad()
def test_paged_state_slots_and_untouched_rows():
    _, causal_conv1d_update = _load_api()
    torch.manual_seed(1)
    n_rows, n_channels, n_slots = 3, 257, 7
    x = torch.randn(n_rows, n_channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n_channels, 4, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(n_slots, n_channels, 4, device="cuda", dtype=torch.bfloat16)
    state_indices = torch.tensor([6, 1, 4], device="cuda", dtype=torch.int32)
    expected_output, expected_state = _reference_step(x, weight, state, state_indices)

    output = causal_conv1d_update(x, state, weight, conv_state_indices=state_indices)

    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(state, expected_state)
    untouched = torch.tensor([0, 2, 3, 5], device="cuda", dtype=torch.long)
    _assert_state_bits_equal(
        state.index_select(0, untouched),
        expected_state.index_select(0, untouched),
    )


@torch.no_grad()
@pytest.mark.parametrize("state_len", [3, 4], ids=["w-minus-one", "legacy-four"])
def test_padding_state_indices_write_zero_and_do_not_mutate_state(state_len):
    _, causal_conv1d_update = _load_api()
    torch.manual_seed(5)
    n_rows, n_channels, n_slots = 4, 257, 5
    x = torch.randn(n_rows, n_channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n_channels, 4, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n_channels, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(n_slots, n_channels, state_len, device="cuda", dtype=torch.bfloat16)
    state_indices = torch.tensor([-1, 3, -1, 1], device="cuda", dtype=torch.int32)
    expected_output, expected_state = _reference_step(
        x,
        weight,
        state,
        state_indices,
        bias=bias,
        activation="silu",
    )

    output = causal_conv1d_update(
        x,
        state,
        weight,
        bias,
        activation="silu",
        conv_state_indices=state_indices,
    )

    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(output[[0, 2]], torch.zeros_like(output[[0, 2]]), rtol=0, atol=0)
    _assert_state_bits_equal(state, expected_state)


@torch.no_grad()
def test_wminus1_state_handoff_from_prefill_matches_next_causal_output():
    _, causal_conv1d_update = _load_api()
    torch.manual_seed(7)
    n_rows, n_channels, prefix_len = 3, 259, 11
    prefix = torch.randn(n_rows, n_channels, prefix_len, device="cuda", dtype=torch.bfloat16)
    next_x = torch.randn(n_rows, n_channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n_channels, 4, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n_channels, device="cuda", dtype=torch.bfloat16)

    # Bulk prefill returns exactly W - 1 history elements. Decode must consume
    # that state without padding it to the legacy four-element cache layout.
    state = prefix[..., -3:].contiguous()
    full_history = torch.cat((prefix, next_x.unsqueeze(-1)), dim=-1)
    expected_window = full_history[..., -4:]
    expected_output = F.silu((expected_window.float() * weight.float().unsqueeze(0)).sum(dim=-1) + bias.float()).to(torch.bfloat16)
    expected_state = full_history[..., -3:].contiguous()

    output = causal_conv1d_update(
        next_x,
        state,
        weight,
        bias,
        activation="silu",
    )

    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(state, expected_state)


@torch.no_grad()
def test_public_api_returns_an_ordinary_tensor():
    _, causal_conv1d_update = _load_api()
    torch.manual_seed(11)
    n_rows, n_channels = 2, 259
    x = torch.randn(n_rows, n_channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n_channels, 4, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(n_rows, n_channels, 4, device="cuda", dtype=torch.bfloat16)
    expected_output, expected_state = _reference_step(x, weight, state)

    result = causal_conv1d_update(x, state, weight)

    assert type(result) is torch.Tensor
    torch.testing.assert_close(result, expected_output, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(state, expected_state)


@torch.no_grad()
@pytest.mark.parametrize("state_len", [3, 4], ids=["w-minus-one", "legacy-four"])
def test_state_shift_and_append_are_bitwise(state_len):
    _, causal_conv1d_update = _load_api()
    torch.manual_seed(2)
    n_rows, n_channels = 2, 257

    # Exercise raw BF16 payloads rather than only finite values.  The output is
    # intentionally ignored; this test proves the mutable cache path is a
    # bitwise shift/append, including signed zero and NaN payloads.
    state_bits = torch.randint(
        -(2**15),
        2**15,
        (n_rows, n_channels, state_len),
        device="cuda",
        dtype=torch.int16,
    )
    x_bits = torch.randint(
        -(2**15),
        2**15,
        (n_rows, n_channels),
        device="cuda",
        dtype=torch.int16,
    )
    state = state_bits.view(torch.bfloat16)
    x = x_bits.view(torch.bfloat16)
    weight = torch.zeros(n_channels, 4, device="cuda", dtype=torch.bfloat16)
    expected_bits = torch.cat((state_bits, x_bits.unsqueeze(-1)), dim=-1)[..., -state_len:]

    causal_conv1d_update(x, state, weight)

    torch.testing.assert_close(state.view(torch.int16), expected_bits, rtol=0, atol=0)


@torch.no_grad()
def test_silu_special_values_and_tails():
    _, causal_conv1d_update = _load_api()
    values = torch.tensor(
        [
            -float("inf"),
            -100.0,
            -20.0,
            -10.0,
            -1.0,
            -0.0,
            0.0,
            1.0,
            10.0,
            20.0,
            100.0,
            float("inf"),
            float("nan"),
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    x = values.unsqueeze(0)
    state = torch.zeros(1, values.numel(), 4, device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(values.numel(), 4, device="cuda", dtype=torch.bfloat16)
    weight[:, -1] = 1
    expected, expected_state = _reference_step(x, weight, state, activation="silu")

    output = causal_conv1d_update(x, state, weight, activation="silu")

    torch.testing.assert_close(output, expected, atol=3e-2, rtol=3e-2, equal_nan=True)
    _assert_state_bits_equal(state, expected_state)
    # Signed zero is determined by the four-term convolution reduction, not by
    # applying SiLU directly to the final x lane.  Compare the observable
    # reduction result bitwise for both zero inputs.
    zero_lanes = torch.tensor([5, 6], device="cuda")
    torch.testing.assert_close(
        output.view(torch.int16).index_select(1, zero_lanes),
        expected.view(torch.int16).index_select(1, zero_lanes),
        rtol=0,
        atol=0,
    )


@torch.no_grad()
def test_public_custom_op_registration_contract():
    from cudnn.ops._causal_conv1d_update import _causal_conv1d_update_primitive

    torch.manual_seed(31)
    n_rows, n_channels, n_slots = 2, 257, 5
    x = torch.randn(n_rows, n_channels, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(n_slots, n_channels, 4, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n_channels, 4, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n_channels, device="cuda", dtype=torch.bfloat16)
    state_indices = torch.tensor([4, 1], device="cuda", dtype=torch.int32)
    test_utils = (
        "test_schema",
        "test_faketensor",
        "test_aot_dispatch_dynamic",
    )

    results = torch.library.opcheck(
        _causal_conv1d_update_primitive,
        (x, state, weight, bias, "silu", None, state_indices),
        test_utils=test_utils,
    )

    assert results == {test: "SUCCESS" for test in test_utils}


@torch.no_grad()
def test_public_plain_cuda_tensors_use_validated_eager_fast_path(monkeypatch):
    ops_module = importlib.import_module("cudnn.ops._causal_conv1d_update")
    x = torch.zeros(2, 257, device="cuda", dtype=torch.bfloat16)
    state = torch.zeros(2, 257, 4, device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(257, 4, device="cuda", dtype=torch.bfloat16)
    sentinel = torch.empty_like(x)
    calls = []

    def validated_native(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(ops_module, "_validated_native_update", validated_native)

    output = ops_module.causal_conv1d_update(x, state, weight, activation="swish")

    assert output is sentinel
    assert calls == [(x, state, weight, None, "silu", None, None)]


@torch.no_grad()
def test_public_eager_fast_path_declines_active_torch_dispatch_mode():
    from torch.utils._python_dispatch import TorchDispatchMode

    ops_module = importlib.import_module("cudnn.ops._causal_conv1d_update")
    x = torch.zeros(2, 257, device="cuda", dtype=torch.bfloat16)
    state = torch.zeros(2, 257, 4, device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(257, 4, device="cuda", dtype=torch.bfloat16)

    class PassthroughMode(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            return function(*args, **(kwargs or {}))

    assert ops_module._can_use_eager_native_fast_path(x, state, weight, None, None, None)
    with PassthroughMode():
        assert not ops_module._can_use_eager_native_fast_path(x, state, weight, None, None, None)


@torch.no_grad()
def test_public_torch_compile_fullgraph_observes_state_mutation():
    _, causal_conv1d_update = _load_api()
    torch.manual_seed(37)
    n_rows, n_channels = 2, 257
    x = torch.randn(n_rows, n_channels, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(n_rows, n_channels, 4, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n_channels, 4, device="cuda", dtype=torch.bfloat16)
    expected_output, expected_state = _reference_step(x, weight, state, activation="silu")

    compiled_update = torch.compile(causal_conv1d_update, fullgraph=True)
    output = compiled_update(x, state, weight, activation="silu")

    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(state, expected_state)


@torch.no_grad()
def test_public_cuda_graph_capture_and_replay_observe_state_mutation():
    _, causal_conv1d_update = _load_api()
    torch.manual_seed(43)
    n_rows, n_channels = 2, 257
    x = torch.randn(n_rows, n_channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n_channels, 4, device="cuda", dtype=torch.bfloat16)
    initial_state = torch.randn(n_rows, n_channels, 4, device="cuda", dtype=torch.bfloat16)

    # Compile and cache the exact native specialization before capture.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        causal_conv1d_update(x, initial_state.clone(), weight, activation="silu")
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    state = initial_state.clone()
    expected_output, expected_state = _reference_step(x, weight, initial_state, activation="silu")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = causal_conv1d_update(x, state, weight, activation="silu")

    # Capture records the launch; only replay makes its output and mutation
    # observable. Poison the graph-owned output to prove replay overwrites it.
    captured_output.fill_(float("nan"))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured_output, expected_output, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(state, expected_state)

    expected_output_2, expected_state_2 = _reference_step(x, weight, expected_state, activation="silu")
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured_output, expected_output_2, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(state, expected_state_2)


@torch.no_grad()
def test_execute_respects_current_torch_stream():
    plan_cls, _ = _load_api()
    torch.manual_seed(3)
    n_rows, n_channels = 2, 1024
    x_real = torch.randn(n_rows, n_channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n_channels, 4, device="cuda", dtype=torch.bfloat16)
    state_real = torch.randn(n_rows, n_channels, 4, device="cuda", dtype=torch.bfloat16)
    expected_output, expected_state = _reference_step(x_real, weight, state_real)

    x = torch.full_like(x_real, float("nan"))
    state = torch.full_like(state_real, float("nan"))
    output = torch.empty_like(x)
    api = plan_cls(x, weight, state, output)
    assert api.check_support()
    api.compile()
    torch.cuda.synchronize()

    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        # Make wrong-stream execution reliably overtake the real input copies.
        torch.cuda._sleep(100_000_000)
        x.copy_(x_real)
        state.copy_(state_real)
        api.execute(x, weight, state, output)
    side.synchronize()

    assert not torch.isnan(output).any(), "kernel read poisoned input from the wrong stream"
    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(state, expected_state)


@torch.no_grad()
def test_private_allocating_route_respects_explicit_cuda_stream():
    from cudnn.causal_conv1d_update_sm100 import _causal_conv1d_update

    torch.manual_seed(13)
    n_rows, n_channels = 2, 1024
    x_real = torch.randn(n_rows, n_channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n_channels, 4, device="cuda", dtype=torch.bfloat16)
    state_real = torch.randn(n_rows, n_channels, 4, device="cuda", dtype=torch.bfloat16)
    expected_output, expected_state = _reference_step(x_real, weight, state_real)

    # Warm the exact signature so this test isolates stream ordering rather
    # than JIT compilation behavior.
    warm_state = state_real.clone()
    _causal_conv1d_update(x_real, warm_state, weight)
    torch.cuda.synchronize()

    x = torch.full_like(x_real, float("nan"))
    state = torch.full_like(state_real, float("nan"))
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(100_000_000)
        x.copy_(x_real)
        state.copy_(state_real)

    result = _causal_conv1d_update(
        x,
        state,
        weight,
        current_stream=cuda.CUstream(side.cuda_stream),
    )
    side.synchronize()

    assert not torch.isnan(result).any(), "kernel read poisoned input from the wrong stream"
    torch.testing.assert_close(result, expected_output, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(state, expected_state)


_DEVICE_ASSERT_WORKER = r"""
import os
import sys

import torch
import cudnn

case = sys.argv[1]
source_cudnn = sys.argv[2]
cudnn.__path__.insert(0, source_cudnn)

from cudnn.causal_conv1d_update_sm100 import _CausalConv1dUpdatePlan

torch.manual_seed(17)
n_rows, n_channels, n_slots = 2, 257, 3
x = torch.randn(n_rows, n_channels, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(n_channels, 4, device="cuda", dtype=torch.bfloat16)
state = torch.randn(n_slots, n_channels, 4, device="cuda", dtype=torch.bfloat16)
output = torch.empty_like(x)
valid_indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
api = _CausalConv1dUpdatePlan(x, weight, state, output, valid_indices)
api.check_support()
api.compile()
api.execute(x, weight, state, output, valid_indices)
torch.cuda.synchronize()

invalid_indices = {
    "below_padding": [-2, 1],
    "out_of_range": [0, n_slots],
    "duplicate": [1, 1],
}[case]
invalid_indices = torch.tensor(invalid_indices, device="cuda", dtype=torch.int32)

try:
    api.execute(x, weight, state, output, invalid_indices)
    torch.cuda.synchronize()
except Exception as error:
    print(f"EXPECTED_DEVICE_FAILURE:{case}:{type(error).__name__}:{error}", flush=True)
    os._exit(0)

print(f"MISSING_DEVICE_FAILURE:{case}", flush=True)
os._exit(9)
"""


@pytest.mark.L1
@pytest.mark.parametrize("case", ["below_padding", "out_of_range", "duplicate"])
def test_invalid_state_indices_fail_closed_in_fresh_process(case):
    # A CUDA device assert poisons its process context.  Keep each contract
    # case isolated and first prove the same compiled object launches validly.
    source_cudnn = Path(__file__).resolve().parents[4] / "python" / "cudnn"
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", _DEVICE_ASSERT_WORKER, case, str(source_cudnn)],
        cwd=Path(__file__).resolve().parents[4],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    diagnostics = result.stdout + result.stderr
    assert result.returncode == 0, diagnostics
    assert f"EXPECTED_DEVICE_FAILURE:{case}:" in diagnostics, diagnostics


@torch.no_grad()
@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda x, w, s, i: (x, w[:, :3], s, i), "Weight tensor shape mismatch"),
        (
            lambda x, w, s, i: (
                torch.empty(
                    (x.shape[0], x.shape[1] * 2),
                    device=x.device,
                    dtype=x.dtype,
                )[:, ::2],
                w,
                s,
                i,
            ),
            r"X must have row-major strides \(ld, 1\)",
        ),
        (lambda x, w, s, i: (x.float(), w, s, i), "X dtype mismatch"),
        (lambda x, w, s, i: (x, w, s[:1], None), "State needs at least N slots"),
        (
            lambda x, w, s, i: (x, w, s, i.to(torch.int64)),
            "State indices dtype mismatch",
        ),
    ],
    ids=["kernel-width", "shape-mismatch", "dtype", "too-few-slots", "index-dtype"],
)
def test_bad_host_contracts_fail_closed(mutate, match):
    plan_cls, _ = _load_api()
    n_rows, n_channels = 2, 8
    x = torch.zeros(n_rows, n_channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(n_channels, 4, device="cuda", dtype=torch.bfloat16)
    state = torch.zeros(n_rows, n_channels, 4, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(n_rows, device="cuda", dtype=torch.int32)
    x, weight, state, indices = mutate(x, weight, state, indices)
    output = torch.empty_like(x)

    api = plan_cls(x, weight, state, output, indices)
    with pytest.raises((TypeError, ValueError), match=match):
        api.check_support()
