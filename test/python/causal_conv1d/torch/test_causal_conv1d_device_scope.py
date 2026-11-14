# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch

import cudnn

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("variant", ["plain", "nwh", "b2b", "medium_fft", "long_fft"])
@pytest.mark.parametrize("fail_binding", [None, "forward", "backward"])
def test_causal_conv1d_preserves_device_and_operand_stream(variant, fail_binding, monkeypatch):
    """Exercise native calls on the operand's side stream with a different caller GPU.

    Intercept the binding before launching so a regression cannot send operand
    pointers to the other GPU's stream. Also check restoration when native calls raise.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    minimum = 92600 if "fft" in variant else 92400
    if cudnn.backend_version() < minimum:
        pytest.skip(f"Requires cuDNN {minimum}")
    importlib.import_module("cudnn.ops.causal_conv1d")
    importlib.import_module("cudnn.ops.fft_causal_conv1d")
    ops = torch.ops.cudnn
    with torch.cuda.device(0):
        torch.manual_seed(17)
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):

            def rand(*shape):
                return 0.1 * torch.randn(*shape, device="cuda:0", dtype=torch.float32)

            if variant == "b2b":
                x, wp, wm, bias, dy = rand(1, 24, 128), rand(24, 4), rand(8, 7), rand(8), rand(1, 8, 128)
                prefix = "b2b_causal_conv1d"
                fwd = lambda: list(ops.b2b_causal_conv1d_fwd_primitive(x, wp, wm, bias))
                bwd = lambda out: list(ops.b2b_causal_conv1d_bwd_primitive(dy, x, wp, wm, bias, out[0]))
            elif variant in ("plain", "nwh"):
                nwh = variant == "nwh"
                shape = (1, 128, 8) if nwh else (1, 8, 128)
                x, weight, bias, dy = rand(*shape), rand(*((4, 8) if nwh else (8, 4))), rand(8), rand(*shape)
                prefix = "causal_conv1d_nwh" if nwh else "causal_conv1d"
                fwd_op, bwd_op = getattr(ops, prefix + "_fwd_primitive"), getattr(ops, prefix + "_bwd_primitive")
                fwd = lambda: [fwd_op(x, weight, bias, "silu")]
                bwd = lambda out: list(bwd_op(dy, x, weight, bias, "silu"))
            elif variant == "medium_fft":
                x, weight, dy = rand(1, 2, 256), rand(2, 128), rand(1, 2, 256)
                prefix = "fft_causal_conv1d"
                fwd = lambda: [ops.fft_causal_conv1d_fwd_primitive(x, weight)]
                bwd = lambda out: list(ops.fft_causal_conv1d_bwd_primitive(dy, x, weight))
            else:
                x, weight, dy = rand(1, 1, 8192), rand(1, 8192), rand(1, 1, 8192)
                prefix = "long_fft_causal_conv1d"
                fwd = lambda: list(ops.long_fft_causal_conv1d_fwd_primitive(x, weight))
                bwd = lambda out: list(ops.long_fft_causal_conv1d_bwd_primitive(dy, x, weight, out[1]))

            expected_out = fwd()
            expected_grad = bwd(expected_out)
            calls = []

            class BindingFailure(RuntimeError):
                pass

            def checked(binding, direction):
                def invoke(stream, *args):
                    assert stream == side.cuda_stream, "binding received another GPU's stream"
                    assert torch.cuda.current_device() == x.device.index
                    calls.append(direction)
                    if fail_binding == direction:
                        raise BindingFailure("injected native binding failure")
                    return binding(stream, *args)

                return invoke

            for direction in ("forward", "backward"):
                name = prefix + "_" + direction
                monkeypatch.setattr(cudnn, name, checked(getattr(cudnn, name), direction))

            torch.cuda.set_device(1)
            caller_stream = torch.cuda.current_stream(1)
            try:
                if fail_binding == "forward":
                    with pytest.raises(BindingFailure):
                        fwd()
                else:
                    actual_out = fwd()
                    assert torch.cuda.current_device() == 1
                    if fail_binding == "backward":
                        with pytest.raises(BindingFailure):
                            bwd(actual_out)
                    else:
                        actual_grad = bwd(actual_out)
                assert torch.cuda.current_device() == 1
                assert torch.cuda.current_stream(1) == caller_stream
                assert torch.cuda.current_stream(0) == side
            finally:
                torch.cuda.set_device(0)
            assert calls == (["forward"] if fail_binding == "forward" else ["forward", "backward"])
            if fail_binding is None:
                side.synchronize()
                torch.testing.assert_close(actual_out[0], expected_out[0], atol=0, rtol=0)
                for actual, expected in zip(actual_grad, expected_grad, strict=True):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
