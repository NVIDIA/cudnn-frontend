# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical and capture checks for the optional shared caller conversions."""

import pytest
import torch

pytest.importorskip("triton")
from causal_c96_layout import pack, unpack_bias

pytestmark = [pytest.mark.L1, pytest.mark.skipif(torch.cuda.get_device_capability() != (10, 0), reason="qualified on SM100")]


def _equal_values(a, b):
    torch.testing.assert_close(a, b, atol=0, rtol=0, equal_nan=True)
    finite = torch.isfinite(a) & torch.isfinite(b)
    assert torch.equal(a.view(torch.int16)[finite], b.view(torch.int16)[finite])


@pytest.mark.parametrize("n", (1, 2))
@pytest.mark.parametrize("d", (1, 4))
@pytest.mark.parametrize("hd", (0, 1, 2))
@pytest.mark.parametrize("k", (96, 192))
def test_frost_c96_layout_capture_and_strides(n, d, hd, k):
    # Noncompact strides and tails exercise both axes of the transpose.
    x = torch.randn(n, 96, d + 2, 7, 18, device="cuda", dtype=torch.bfloat16)[:, :, :d, :, ::2]
    history = None if hd == 0 else torch.randn(n, 96, hd + 1, 7, 18, device=x.device, dtype=x.dtype)[:, :, :hd, :, ::2]
    y = torch.empty((n, k, d, 7, 9), dtype=x.dtype, device=x.device, memory_format=torch.channels_last_3d).normal_()
    bias = torch.randn(k, dtype=x.dtype, device=x.device)

    def run():
        return pack(x, history), unpack_bias(y, bias)

    def verify(outputs):
        a, b = outputs
        expected_x = x if history is None else torch.cat((history, x), dim=2)
        assert a.is_contiguous(memory_format=torch.channels_last_3d)
        assert b.is_contiguous()
        assert torch.equal(a.view(torch.int16), expected_x.view(torch.int16))
        _equal_values(b, y + bias.view(1, -1, 1, 1, 1))

    verify(run())
    captured = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    try:
        with torch.cuda.graph(captured, stream=stream):
            outputs = run()
        for exceptional in (False, True, False):
            x.normal_()
            y.normal_()
            bias.normal_()
            if history is not None:
                history.normal_()
            if exceptional:
                x[0, 0, 0, 0, :4] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0], device=x.device, dtype=x.dtype)
                if history is not None:
                    history[0, 0, 0, 0, 0] = float("nan")
                y[0, :4, 0, 0, 0] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0], device=y.device, dtype=y.dtype)
                bias[:4] = torch.tensor([0.0, -float("inf"), float("inf"), -0.0], device=y.device, dtype=y.dtype)
            captured.replay()
            torch.cuda.synchronize()
            verify(outputs)
    finally:
        captured.reset()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        outputs = run()
    torch.cuda.current_stream().wait_stream(stream)
    verify(outputs)


def test_frost_c96_layout_rejects_invalid_operands():
    x = torch.zeros((1, 96, 1, 7, 9), device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x, memory_format=torch.channels_last_3d)
    bias = torch.zeros(96, device=x.device, dtype=x.dtype)
    for bad_x, history in ((x.float(), None), (x[:, :64], None), (x, x[:, :, :0]), (x, x[:, :95]), (x, x[0])):
        with pytest.raises(ValueError):
            pack(bad_x, history)
    with pytest.raises(ValueError, match="batch size"):
        pack(torch.empty((65536, 96, 1, 1, 1), device=x.device, dtype=x.dtype), None)
    for bad_y, bad_bias in ((x, bias), (y, bias.float()), (y, bias[::2]), (y, bias.view(1, -1))):
        with pytest.raises(ValueError):
            unpack_bias(bad_y, bad_bias)
