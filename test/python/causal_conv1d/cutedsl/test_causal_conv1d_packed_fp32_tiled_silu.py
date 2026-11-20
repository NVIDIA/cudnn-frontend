# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Long FP32 SiLU gradients with a contiguous gradient into packed backward."""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_replays, snapshot
from test_causal_conv1d_packed import _dsl
from test_causal_conv1d_packed_fp32 import _check_reference

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def deterministic_algorithms():
    _dsl()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _run_contiguous_gradient(x, weight, bias, initial, lengths, dy, dfinal):
    from functools import partial
    from cudnn.ops import causal_conv1d as _causal_conv1d

    causal_conv1d = partial(_causal_conv1d, backend="se")

    xx, ww, bb, ii = (value.detach().clone().requires_grad_() for value in (x, weight, bias, initial))
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), device=x.device, dtype=torch.int32)
    y_bht, final_shk = causal_conv1d(
        xx.T.unsqueeze(0),
        ww,
        bb,
        cu_seqlens=cu,
        initial_states=ii.transpose(1, 2),
        activation="silu",
        return_final_states=True,
    )
    # Squeeze preserves the gradient view; tensor indexing would materialize
    # a channel-major gradient through SelectBackward and exercise fallback.
    y, final = y_bht.squeeze(0).T, final_shk.transpose(1, 2)
    gradients = torch.autograd.grad((y, final), (xx, ww, bb, ii), (dy, dfinal))
    actual = dict(y=y, final=final)
    actual.update(zip(("dx", "dw", "db", "di"), gradients))
    return actual


@pytest.mark.parametrize("seed", [1907, 319])
@pytest.mark.parametrize("width", [2, 4, 8])
@pytest.mark.parametrize("lengths", [[0, 1, 2, 7, 65, 4097, 0], [0, 1, 7, 8192, 8211, 0]], ids=["T4172", "T16411"])
@pytest.mark.parametrize("layout", ["HK", "KH"])
def test_long_fp32_silu_contiguous_output_gradient(monkeypatch, seed, width, lengths, layout):
    from cudnn.causal_conv1d_se import packed as adapter

    torch.manual_seed(seed)
    hidden = 136
    x = torch.randn(sum(lengths), hidden, device="cuda")
    weight = torch.randn(hidden, width, device="cuda")
    if layout == "KH":
        weight = weight.T.contiguous().T
    bias = torch.randn(hidden, device="cuda")
    initial = torch.randn(len(lengths), width - 1, hidden, device="cuda")
    dy, dfinal = torch.randn_like(x), torch.randn_like(initial)
    inputs = dict(x=x, weight=weight, bias=bias, initial=initial, dy=dy, dfinal=dfinal)
    before = snapshot(inputs)
    original_launch = adapter._launch
    backward_strides = []

    def launch(direction, tensors, kernel_width, silu):
        if direction == "backward":
            assert kernel_width == width and silu
            observed = tuple(tuple(tensors[index].stride()) for index in (0, 5, 8))
            assert observed == ((hidden, 1),) * 3, "The test must exercise contiguous packed backward gradients"
            backward_strides.append(observed)
        return original_launch(direction, tensors, kernel_width, silu)

    monkeypatch.setattr(adapter, "_launch", launch)
    actual = assert_replays(lambda: _run_contiguous_gradient(x, weight, bias, initial, lengths, dy, dfinal), repeats=3)
    assert len(backward_strides) == 3
    _check_reference(actual, x, weight, bias, initial, lengths, dy, dfinal, silu=True)
    assert_bitwise(inputs, before, "contiguous-gradient inputs mutated")
