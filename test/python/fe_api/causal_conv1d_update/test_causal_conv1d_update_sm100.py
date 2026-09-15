# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU correctness for the public causal-convolution decode update."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from cudnn._causal_conv1d_arch import (
    is_supported_causal_conv1d_update_compute_capability,
)


def _has_supported_gpu() -> bool:
    return torch.cuda.is_available() and is_supported_causal_conv1d_update_compute_capability(torch.cuda.get_device_capability())


def _has_supported_cutedsl() -> bool:
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    return installed and not cutedsl_too_old(version)


pytestmark = [
    pytest.mark.skipif(
        not _has_supported_gpu(),
        reason="requires a functionally supported GPU architecture",
    ),
    pytest.mark.skipif(
        not _has_supported_cutedsl(),
        reason="requires nvidia-cutlass-dsl at or above cudnn.frost.buffers.CUTEDSL_MIN_VERSION",
    ),
]


def _load_api():
    try:
        from cudnn.ops import causal_conv1d_update
    except ImportError as error:
        pytest.skip(f"CuTe DSL dependencies unavailable: {error}")
    return causal_conv1d_update


def _reference_step(
    x,
    weight,
    state,
    state_indices=None,
    bias=None,
    activation=None,
):
    """Explicit FP32 output oracle plus exact BF16 state transition."""

    rows = x.shape[0]
    slots = torch.arange(rows, device=x.device, dtype=torch.long) if state_indices is None else state_indices.long()
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
    torch.testing.assert_close(
        actual.contiguous().view(torch.int16),
        expected.contiguous().view(torch.int16),
        rtol=0,
        atol=0,
    )


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


@pytest.mark.L0
@torch.no_grad()
def test_nonzero_state_across_consecutive_steps():
    causal_conv1d_update = _load_api()
    torch.manual_seed(0)
    rows, channels = 3, 521
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(channels, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(rows, channels, 4, device="cuda", dtype=torch.bfloat16)

    for _ in range(3):
        x = torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16)
        expected_output, expected_state = _reference_step(x, weight, state, bias=bias, activation="silu")
        output = causal_conv1d_update(x, state, weight, bias, activation="silu")
        torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
        _assert_state_bits_equal(state, expected_state)


@pytest.mark.L0
@torch.no_grad()
@pytest.mark.parametrize("state_len", [3, 4], ids=["w-minus-one", "legacy-four"])
def test_padded_x_rows_match_reference(state_len):
    causal_conv1d_update = _load_api()
    torch.manual_seed(11 + state_len)
    rows, channels, row_stride = 3, 10, 16
    dense_x = torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16)
    x = _padded_x(dense_x, row_stride)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(rows, channels, state_len, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(channels, device="cuda", dtype=torch.bfloat16)
    expected_output, expected_state = _reference_step(dense_x, weight, state, bias=bias, activation="silu")

    output = causal_conv1d_update(x, state, weight, bias, activation="silu")

    assert output.shape == dense_x.shape
    assert output.dtype == dense_x.dtype
    assert output.device == dense_x.device
    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(state, expected_state)


@pytest.mark.L0
@torch.no_grad()
@pytest.mark.parametrize("channels", [2048, 4096])
def test_n128_representative_shape_correctness(channels):
    causal_conv1d_update = _load_api()
    torch.manual_seed(1)
    rows = 128
    x = torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(rows, channels, 4, device="cuda", dtype=torch.bfloat16)
    expected_output, expected_state = _reference_step(x, weight, state)

    output = causal_conv1d_update(x, state, weight)

    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(state, expected_state)


@pytest.mark.L0
@torch.no_grad()
def test_paged_state_slots_and_untouched_rows():
    causal_conv1d_update = _load_api()
    torch.manual_seed(4)
    rows, slots, channels = 2, 4, 259
    x = torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(slots, channels, 4, device="cuda", dtype=torch.bfloat16)
    state_indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    expected_output, expected_state = _reference_step(x, weight, state, state_indices)

    output = causal_conv1d_update(x, state, weight, conv_state_indices=state_indices)

    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(state, expected_state)


@pytest.mark.L0
@torch.no_grad()
@pytest.mark.parametrize("state_len", [3, 4], ids=["w-minus-one", "legacy-four"])
def test_padding_state_indices_write_zero_and_do_not_mutate_state(state_len):
    causal_conv1d_update = _load_api()
    torch.manual_seed(5 + state_len)
    rows, slots, channels = 4, 5, 257
    x = torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(channels, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(slots, channels, state_len, device="cuda", dtype=torch.bfloat16)
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


@pytest.mark.L0
@torch.no_grad()
def test_channel_fast_wminus1_state_matches_next_causal_output():
    causal_conv1d_update = _load_api()
    torch.manual_seed(7)
    rows, channels, prefix_len = 3, 259, 11
    prefix = torch.randn(rows, channels, prefix_len, device="cuda", dtype=torch.bfloat16)
    next_x = torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(channels, device="cuda", dtype=torch.bfloat16)
    state = prefix[..., -3:].transpose(1, 2).contiguous().transpose(1, 2)
    full_history = torch.cat((prefix, next_x.unsqueeze(-1)), dim=-1)
    expected_window = full_history[..., -4:]
    expected_output = F.silu((expected_window.float() * weight.float().unsqueeze(0)).sum(dim=-1) + bias.float()).to(torch.bfloat16)
    expected_state = full_history[..., -3:]

    output = causal_conv1d_update(next_x, state, weight, bias, activation="silu")

    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(state, expected_state)


@pytest.mark.L0
@torch.no_grad()
def test_public_api_returns_an_ordinary_tensor():
    causal_conv1d_update = _load_api()
    rows, channels = 2, 257
    x = torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(rows, channels, 4, device="cuda", dtype=torch.bfloat16)
    expected_output, expected_state = _reference_step(x, weight, state)

    result = causal_conv1d_update(x, state, weight)

    assert type(result) is torch.Tensor
    torch.testing.assert_close(result, expected_output, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(state, expected_state)


@pytest.mark.L0
@torch.no_grad()
@pytest.mark.parametrize("state_len", [3, 4], ids=["w-minus-one", "legacy-four"])
def test_state_shift_and_append_are_bitwise(state_len):
    causal_conv1d_update = _load_api()
    torch.manual_seed(2)
    rows, channels = 2, 257
    state_bits = torch.randint(
        -(2**15),
        2**15,
        (rows, channels, state_len),
        device="cuda",
        dtype=torch.int16,
    )
    x_bits = torch.randint(
        -(2**15),
        2**15,
        (rows, channels),
        device="cuda",
        dtype=torch.int16,
    )
    state = state_bits.view(torch.bfloat16)
    x = x_bits.view(torch.bfloat16)
    weight = torch.zeros(channels, 4, device="cuda", dtype=torch.bfloat16)
    expected_bits = torch.cat((state_bits, x_bits.unsqueeze(-1)), dim=-1)[..., -state_len:]

    causal_conv1d_update(x, state, weight)

    torch.testing.assert_close(state.view(torch.int16), expected_bits, rtol=0, atol=0)


@pytest.mark.L0
@torch.no_grad()
def test_channel_fast_wminus1_state_shift_and_append_are_bitwise():
    causal_conv1d_update = _load_api()
    torch.manual_seed(3)
    rows, channels = 2, 257
    backing_bits = torch.randint(
        -(2**15),
        2**15,
        (rows, 3, channels),
        device="cuda",
        dtype=torch.int16,
    )
    state = backing_bits.view(torch.bfloat16).transpose(1, 2)
    initial_bits = state.contiguous().view(torch.int16)
    x_bits = torch.randint(
        -(2**15),
        2**15,
        (rows, channels),
        device="cuda",
        dtype=torch.int16,
    )
    x = x_bits.view(torch.bfloat16)
    weight = torch.zeros(channels, 4, device="cuda", dtype=torch.bfloat16)
    expected_bits = torch.cat((initial_bits, x_bits.unsqueeze(-1)), dim=-1)[..., -3:]

    causal_conv1d_update(x, state, weight)

    torch.testing.assert_close(state.contiguous().view(torch.int16), expected_bits, rtol=0, atol=0)


@pytest.mark.L0
@torch.no_grad()
def test_silu_special_values_and_channel_tail():
    causal_conv1d_update = _load_api()
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
        dtype=torch.float32,
    ).to(torch.bfloat16)
    channels = 257
    x = values.repeat((channels + values.numel() - 1) // values.numel())[:channels]
    x = x.unsqueeze(0)
    state = torch.zeros(1, channels, 3, device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(channels, 4, device="cuda", dtype=torch.bfloat16)
    weight[:, 3] = 1
    expected = F.silu(x.float()).to(torch.bfloat16)

    output = causal_conv1d_update(x, state, weight, activation="silu")

    finite = torch.isfinite(expected)
    torch.testing.assert_close(output[finite], expected[finite], atol=3e-2, rtol=3e-2)
    infinite = torch.isinf(expected)
    torch.testing.assert_close(output[infinite], expected[infinite], rtol=0, atol=0)
    assert torch.equal(torch.isnan(output), torch.isnan(expected))


@pytest.mark.L0
@torch.no_grad()
def test_silu_and_swish_aliases_are_observably_identical():
    causal_conv1d_update = _load_api()
    torch.manual_seed(13)
    rows, channels = 3, 257
    x = torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    initial_state = torch.randn(rows, channels, 3, device="cuda", dtype=torch.bfloat16)
    silu_state = initial_state.clone()
    swish_state = initial_state.clone()

    silu_output = causal_conv1d_update(x, silu_state, weight, activation="silu")
    swish_output = causal_conv1d_update(x, swish_state, weight, activation="swish")

    torch.testing.assert_close(silu_output, swish_output, rtol=0, atol=0)
    _assert_state_bits_equal(silu_state, swish_state)


@pytest.mark.L0
@torch.no_grad()
def test_public_torch_compile_fullgraph_observes_state_mutation():
    causal_conv1d_update = _load_api()
    torch.manual_seed(23)
    rows, channels = 2, 257
    x = torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(rows, channels, 4, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    expected_output, expected_state = _reference_step(x, weight, state, activation="silu")

    compiled_update = torch.compile(causal_conv1d_update, fullgraph=True)
    output = compiled_update(x, state, weight, activation="silu")

    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(state, expected_state)


@pytest.mark.L0
@torch.no_grad()
def test_public_cuda_graph_capture_and_replay_observe_state_mutation():
    causal_conv1d_update = _load_api()
    torch.manual_seed(29)
    rows, channels = 2, 257
    x = torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    initial_state = torch.randn(rows, channels, 4, device="cuda", dtype=torch.bfloat16)
    warmup_stream = torch.cuda.Stream()
    with torch.cuda.stream(warmup_stream):
        causal_conv1d_update(x, initial_state.clone(), weight, activation="silu")
    warmup_stream.synchronize()

    state = initial_state.clone()
    expected_output, expected_state = _reference_step(x, weight, initial_state, activation="silu")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = causal_conv1d_update(x, state, weight, activation="silu")
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


@pytest.mark.L0
@torch.no_grad()
def test_public_call_respects_current_torch_stream():
    causal_conv1d_update = _load_api()
    torch.manual_seed(31)
    rows, channels = 2, 257
    x_real = torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16)
    state_real = torch.randn(rows, channels, 4, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    expected_output, expected_state = _reference_step(x_real, weight, state_real)
    causal_conv1d_update(x_real, state_real.clone(), weight)
    torch.cuda.synchronize()

    x = torch.full_like(x_real, float("nan"))
    state = torch.full_like(state_real, float("nan"))
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(100_000_000)
        x.copy_(x_real)
        state.copy_(state_real)
        output = causal_conv1d_update(x, state, weight)
    side.synchronize()

    assert not torch.isnan(output).any(), "kernel read poisoned data from another stream"
    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(state, expected_state)


_DEVICE_ASSERT_WORKER = r"""
import os
import sys

import torch
import cudnn

case = sys.argv[1]
source_cudnn = sys.argv[2]
cudnn.__path__.insert(0, source_cudnn)

from cudnn.ops import causal_conv1d_update

torch.manual_seed(17)
rows, channels, slots = 2, 257, 3
x = torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
state = torch.randn(slots, channels, 4, device="cuda", dtype=torch.bfloat16)
valid_indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
causal_conv1d_update(x, state, weight, conv_state_indices=valid_indices)
torch.cuda.synchronize()

invalid_indices = {
    "below_padding": [-2, 1],
    "out_of_range": [0, slots],
    "duplicate": [1, 1],
}[case]
invalid_indices = torch.tensor(invalid_indices, device="cuda", dtype=torch.int32)

try:
    causal_conv1d_update(x, state, weight, conv_state_indices=invalid_indices)
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


@pytest.mark.L0
@torch.no_grad()
@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda x, w, s, i: (x, w[:, :3].contiguous(), s, i), "supports only"),
        (lambda x, w, s, i: (x, w.float(), s, i), "weight must have dtype"),
        (lambda x, w, s, i: (x, w, s[..., :2].contiguous(), i), "L must satisfy"),
        (lambda x, w, s, i: (x, w, s, i[:1]), "must have shape"),
        (
            lambda x, w, s, i: (
                torch.empty_strided(
                    x.shape,
                    (x.shape[1] + 2, 1),
                    device=x.device,
                    dtype=x.dtype,
                ),
                w,
                s,
                i,
            ),
            "16-byte-aligned",
        ),
    ],
)
def test_bad_public_contracts_fail_closed(mutate, match):
    causal_conv1d_update = _load_api()
    rows, channels = 2, 10
    x = torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(rows, channels, 4, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(rows, device="cuda", dtype=torch.int32)
    bad_x, bad_weight, bad_state, bad_indices = mutate(x, weight, state, indices)

    with pytest.raises((TypeError, ValueError, NotImplementedError), match=match):
        causal_conv1d_update(
            bad_x,
            bad_state,
            bad_weight,
            conv_state_indices=bad_indices,
        )
