# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 00: 3D convolution forward with BF16 inputs and outputs.

Exercises multiple input shapes, asymmetric padding, and spatial strides.
Also covers anisotropic dilation; bias is omitted.

Inputs:
    X: input, in bp16, NDHWC format
    K: filter, in bp16, NDHWC format
    Y: output, in bf16, NDHWC format
"""

from __future__ import annotations

import cudnn
import torch

from common import InputShape, build_frost_conv_plans


def _run(shape: InputShape) -> None:
    # Prepare inputs. conv_fprop only support NCDHW inputs with NDHWC layout.
    X_gpu = torch.rand(shape.n, shape.c, shape.d, shape.h, shape.w, dtype=torch.bfloat16, device="cuda").to(memory_format=torch.channels_last_3d)
    K_gpu = torch.rand(shape.k, shape.c, shape.t, shape.r, shape.s, dtype=torch.bfloat16, device="cuda").to(memory_format=torch.channels_last_3d)

    # torch.conv3d only accepts symmetric padding. Pad explicitly so the
    # reference also covers distinct pre/post padding.
    pre_d, pre_h, pre_w = shape.pre_padding
    post_d, post_h, post_w = shape.post_padding
    X_ref = torch.nn.functional.pad(X_gpu, (pre_w, post_w, pre_h, post_h, pre_d, post_d))
    Y_ref = torch.conv3d(X_ref, K_gpu, stride=shape.stride, dilation=shape.dilation)
    Y_actual = torch.empty(Y_ref.shape, dtype=Y_ref.dtype, device="cuda", memory_format=torch.channels_last_3d)

    # Build graph.
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, compute_data_type=cudnn.data_type.FLOAT)
    X = g.tensor_like(X_gpu)
    K = g.tensor_like(K_gpu)
    Y = g.conv_fprop(
        X,
        K,
        name="conv",
        pre_padding=shape.pre_padding,
        post_padding=shape.post_padding,
        stride=shape.stride,
        dilation=shape.dilation,
    )
    Y.set_output(True).set_dim(Y_actual.shape).set_stride(Y_actual.stride())
    build_frost_conv_plans(g)

    # Run with cudnn frontend.
    workspace = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({X: X_gpu, K: K_gpu, Y: Y_actual}, workspace)
    torch.cuda.synchronize()

    torch.testing.assert_close(Y_actual, Y_ref, atol=1e-4, rtol=1e-3)
    print(f"[00] PASS    conv {shape}")


def main():
    # _FpropCTMKernel currently uses a 256x256 implicit-GEMM output tile and a
    # 64-channel GEMM-K tile. Keep M=N*Z*P*Q at 256, K at 256, and C at 64
    # while varying batch/spatial input shapes and convolution geometry.
    shapes = (
        InputShape(n=1, d=6, h=10, w=10, c=64, k=256, t=3, r=3, s=3),
        InputShape(n=2, d=4, h=10, w=10, c=64, k=256, t=3, r=3, s=3),
        InputShape(
            n=1,
            d=5,
            h=9,
            w=8,
            c=64,
            k=256,
            t=3,
            r=3,
            s=3,
            pre_padding=(0, 1, 2),
            post_padding=(1, 0, 0),
        ),
        InputShape(
            n=1,
            d=7,
            h=15,
            w=15,
            c=64,
            k=256,
            t=3,
            r=3,
            s=3,
            pre_padding=(1, 1, 1),
            post_padding=(1, 1, 1),
            stride=(2, 2, 2),
        ),
        InputShape(
            n=1,
            d=4,
            h=8,
            w=8,
            c=64,
            k=256,
            t=3,
            r=3,
            s=3,
            pre_padding=(2, 1, 3),
            post_padding=(2, 1, 3),
            dilation=(2, 1, 3),
        ),
    )

    for shape in shapes:
        _run(shape)


if __name__ == "__main__":
    main()
