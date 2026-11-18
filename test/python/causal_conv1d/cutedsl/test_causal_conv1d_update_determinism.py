# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public FP16 decode output and state determinism. Decode is inference-only."""

import pytest
import torch
from cudnn._causal_conv1d_arch import is_supported_causal_conv1d_update_compute_capability
from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.ops import causal_conv1d_update
from determinism_utils import assert_bitwise, assert_golden, assert_replays
from reference import causal_conv1d_decode_reference

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not is_supported_causal_conv1d_update_compute_capability(torch.cuda.get_device_capability()),
        reason="requires a GPU supported by causal_conv1d_update",
    ),
    pytest.mark.skipif(
        cutedsl_requirement_error("causal_conv1d_update") is not None,
        reason="requires CuTe DSL supported by causal_conv1d_update",
    ),
]

_LAYOUTS = [
    pytest.param(3, False, id="fprop-fp16-L3-compact"),
    pytest.param(3, True, id="fprop-fp16-L3-channel-last"),
    pytest.param(4, False, id="fprop-fp16-L4-compact"),
]


def _operands(rows, channels, slots, state_len, activation):
    # Generate on CPU so the case data do not depend on the GPU's SM count.
    generator = torch.Generator().manual_seed(2026)
    x = torch.randn((rows, channels), generator=generator).to(device="cuda", dtype=torch.float16)
    state = torch.randn((slots, channels, state_len), generator=generator).to(device="cuda", dtype=torch.float16)
    weight = torch.randn((channels, 4), generator=generator).to(device="cuda", dtype=torch.float16)
    bias = torch.randn((channels,), generator=generator).to(device="cuda", dtype=torch.float16) if activation == "silu" else None
    return x, state, weight, bias


def _padded_rows(x):
    rows, channels = x.shape
    storage = x.new_full((rows, ((channels + 7) // 8) * 8), float("nan"))
    storage[:, :channels].copy_(x)
    return storage


def _run_decode(x, state, weight, bias, activation, indices, channel_last):
    if channel_last:
        state = state.transpose(1, 2).contiguous().transpose(1, 2)
    else:
        state = state.clone()
    output = causal_conv1d_update(x, state, weight, bias, activation, conv_state_indices=indices)
    assert output.dtype == x.dtype and output.shape == x.shape and output.is_contiguous()
    return {"output": output, "state": state}


def _check_reference(actual, x, initial, weight, bias, activation, indices):
    output, state = causal_conv1d_decode_reference(x, weight, initial, indices, bias=bias, activation=activation)
    torch.testing.assert_close(actual["output"], output, atol=2e-5, rtol=1e-3)
    assert_bitwise({"state": actual["state"]}, {"state": state}, "reference transition")
    assert_bitwise({"padding": actual["output"][indices == -1]}, {"padding": output[indices == -1]}, "padding output")


@torch.no_grad()
@pytest.mark.parametrize("state_len,channel_last", _LAYOUTS)
@pytest.mark.parametrize("activation", ["identity", "silu"])
def test_fp16_decode_replay_and_batch_invariance(state_len, channel_last, activation):
    # 165 CTAs, including a one-channel tail across the 256-channel tile boundary.
    rows, channels, slots = 33, 1025, 37
    x, initial, weight, bias = _operands(rows, channels, slots, state_len, activation)
    storage = _padded_rows(x)
    indices = torch.arange(rows, device=x.device, dtype=torch.int32)
    indices[0] = -1
    indices[1] = 36
    row, slot = 1, 36
    inputs = {"storage": storage, "initial_state": initial, "weight": weight, "indices": indices}
    if bias is not None:
        inputs["bias"] = bias
    before = {name: tensor.clone() for name, tensor in inputs.items()}

    def run():
        return _run_decode(storage[:, :channels], initial, weight, bias, activation, indices, channel_last)

    repeated = assert_replays(run)
    _check_reference(repeated, x, initial, weight, bias, activation, indices)
    assert_bitwise(inputs, before, "read-only operands")
    target = {name: repeated[name][index] for name, index in (("output", row), ("state", slot))}

    # A sequence can run alone, pack with different neighbors, move to another
    # batch row, or be duplicated into a distinct mutable state slot.
    def check_target(values, state, routing, output_row, state_slot, label):
        result = _run_decode(_padded_rows(values)[:, :channels], state, weight, bias, activation, routing, channel_last)
        assert_bitwise({"output": result["output"][output_row], "state": result["state"][state_slot]}, target, label)

    single_index = indices.new_zeros((1,))
    check_target(x[row : row + 1], initial[slot : slot + 1], single_index, 0, 0, "alone and remapped")
    check_target(x.flip((0,)), initial, indices.flip((0,)), rows - row - 1, slot, "reordered")

    changed_x, changed_state = -x, -initial
    changed_x[row].copy_(x[row])
    changed_state[slot].copy_(initial[slot])
    check_target(changed_x, changed_state, indices, row, slot, "changed neighbors")

    duplicated_x = torch.cat((x, x[row : row + 1]), dim=0)
    duplicated_state = torch.cat((initial, initial[slot : slot + 1]), dim=0)
    duplicated_indices = torch.cat((indices, indices.new_tensor([slots])))
    check_target(duplicated_x, duplicated_state, duplicated_indices, rows, slots, "duplicated into a new slot")


@torch.no_grad()
@pytest.mark.parametrize("state_len,channel_last", _LAYOUTS)
@pytest.mark.parametrize("activation", ["identity", "silu"])
def test_fp16_decode_golden(state_len, channel_last, activation):
    x, initial, weight, bias = _operands(2, 17, 3, state_len, activation)
    indices = torch.tensor([2, -1], device=x.device, dtype=torch.int32)

    def run():
        return _run_decode(x, initial, weight, bias, activation, indices, channel_last)

    actual = assert_replays(run)
    _check_reference(actual, x, initial, weight, bias, activation, indices)
    inputs = {"x": x, "initial_state": initial, "weight": weight, "indices": indices}
    if bias is not None:
        inputs["bias"] = bias
    layout = "channel_last" if channel_last else "compact"
    assert_golden(f"causal_conv1d_update_fp16_{activation}_l{state_len}_{layout}", inputs, actual)
