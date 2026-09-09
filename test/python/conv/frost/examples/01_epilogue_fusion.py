# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 01: 3D convolution forward + epilogue with BF16 inputs and outputs.

Exercises multiple unary epilogue functions.

Inputs:
    X: input, in bf16, NDHWC format
    K: filter, in bf16, NDHWC format
    Y: output, in bf16, NDHWC format
"""

from __future__ import annotations

from collections.abc import Callable

import cudnn
import torch

from common import InputShape, build_frost_conv_plans


_EPILOGUES: dict[str, tuple[Callable, Callable]] = {
    "abs": (lambda g, x: g.abs(x, name="abs"), torch.abs),
    "identity": (lambda g, x: g.identity(x, name="identity"), lambda x: x),
    "neg": (lambda g, x: g.neg(x, name="neg"), torch.neg),
    "relu": (lambda g, x: g.relu(x, name="relu"), torch.relu),
    "leaky_relu": (lambda g, x: g.leaky_relu(x, name="leaky_relu", negative_slope=0.01), lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.01)),
}


def _run(shape: InputShape, epilogue_name: str) -> None:
    graph_epilogue, reference_epilogue = _EPILOGUES[epilogue_name]

    # Center inputs around zero so ReLU/abs exercise both branches.
    X_gpu = torch.randn(shape.n, shape.c, shape.d, shape.h, shape.w, dtype=torch.bfloat16, device="cuda").to(memory_format=torch.channels_last_3d)
    K_gpu = torch.randn(shape.k, shape.c, shape.t, shape.r, shape.s, dtype=torch.bfloat16, device="cuda").to(memory_format=torch.channels_last_3d)

    pre_d, pre_h, pre_w = shape.pre_padding
    post_d, post_h, post_w = shape.post_padding
    X_ref = torch.nn.functional.pad(X_gpu, (pre_w, post_w, pre_h, post_h, pre_d, post_d))
    conv_ref = torch.conv3d(X_ref, K_gpu, stride=shape.stride, dilation=shape.dilation)
    Y_ref = reference_epilogue(conv_ref)

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    X = g.tensor_like(X_gpu)
    K = g.tensor_like(K_gpu)
    conv = g.conv_fprop(
        X,
        K,
        name="conv",
        pre_padding=shape.pre_padding,
        post_padding=shape.post_padding,
        stride=shape.stride,
        dilation=shape.dilation,
    )
    Y = graph_epilogue(g, conv)
    Y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_dim(Y_ref.shape).set_stride(Y_ref.stride())
    build_frost_conv_plans(g)

    Y_actual = torch.empty_like(Y_ref)
    workspace = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({X: X_gpu, K: K_gpu, Y: Y_actual}, workspace)
    torch.cuda.synchronize()

    torch.testing.assert_close(Y_actual, Y_ref, atol=1e-2, rtol=1e-2)
    print(f"[01] PASS    conv + {epilogue_name}")


def main() -> None:
    # Preserve _FpropCTMKernel's exact 256x256 implicit-GEMM output tile and
    # 64-channel GEMM-K tile assumptions while varying the fused epilogue.
    shape = InputShape(n=1, d=6, h=10, w=10, c=64, k=256, t=3, r=3, s=3)
    for epilogue_name in _EPILOGUES:
        _run(shape, epilogue_name)


if __name__ == "__main__":
    main()
