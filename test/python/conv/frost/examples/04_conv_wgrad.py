# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 04: 3D convolution weight gradient (dW) with the FROST wgrad kernel.

Calls ``cudnn/conv/frost/templates/sm100_conv_wgrad.py`` directly -- the kernel the
``frost_conv_wgrad`` graph engine wraps -- instead of going through ``cudnn.pygraph``,
so the whole host contract stays visible:

    dW[k, c, t, r, s] = sum over (n, z, p, q) of
        dY[n, k, z, p, q] * X[n, c, z*sD - padD + t*dilD, ...]

    1. ``compile(ab_dtype, c_dtype, config, epilogue)`` JIT-compiles one artifact
       per (operand dtype, dW dtype, tile config, epilogue). Shapes and the
       convolution geometry are runtime arguments, so one artifact serves every
       convolution of that signature.
    2. The artifact takes the three operands with channels innermost: it reads
       ``dy.shape`` as (n, z, p, q, k), ``dw.shape`` as (k, t, r, s, c) and
       ``x.shape`` as (n, d, h, w, c), plus the stream and the four
       (pre-padding, post-padding, stride, dilation) triplets.
    3. The epilogue runs on the FP32 accumulator before the store, so dW can be
       emitted as BF16 or FP32 and can fuse a pointwise op.

Runs the kernel for BF16 and FP32 operands, with and without a fused ReLU, and
checks every variant against PyTorch's own convolution weight gradient
(``torch.nn.grad.conv3d_weight``) evaluated in FP64.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class ConvGeometry:
    """A 3D convolution: X is (n, c, d, h, w), the filter is (k, c, t, r, s)."""

    n: int
    c: int
    d: int
    h: int
    w: int
    k: int
    t: int
    r: int
    s: int
    pre_padding: tuple[int, int, int] = (0, 0, 0)
    post_padding: tuple[int, int, int] | None = None
    stride: tuple[int, int, int] = (1, 1, 1)
    dilation: tuple[int, int, int] = (1, 1, 1)

    @property
    def padding(self) -> tuple[int, int, int]:
        return self.pre_padding

    @property
    def padded_end(self) -> tuple[int, int, int]:
        """Padding after the last filtered input element; defaults to ``pre_padding``."""
        return self.pre_padding if self.post_padding is None else self.post_padding

    @property
    def x_shape(self) -> tuple[int, ...]:
        return (self.n, self.c, self.d, self.h, self.w)

    @property
    def w_shape(self) -> tuple[int, ...]:
        """dW is filter-shaped: (k, c) channels times the (t, r, s) filter positions."""
        return (self.k, self.c, self.t, self.r, self.s)

    @property
    def dy_shape(self) -> tuple[int, ...]:
        """dY is the forward output's shape: (n, k, z, p, q)."""
        out = tuple(
            (extent + pre + post - (filter_extent - 1) * dil - 1) // st + 1
            for extent, pre, post, filter_extent, st, dil in zip(
                (self.d, self.h, self.w), self.pre_padding, self.padded_end, (self.t, self.r, self.s), self.stride, self.dilation
            )
        )
        return (self.n, self.k, *out)


def _channels_last_3d(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Empty storage with the channels-innermost compact stride the kernel requires."""
    return torch.empty(shape, dtype=dtype, device="cuda", memory_format=torch.channels_last_3d)


def _filled(shape: tuple[int, ...], dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = torch.randn(shape, dtype=torch.float32, device="cuda", generator=generator) * 0.25
    return _channels_last_3d(shape, dtype).copy_(values)


# A fused epilogue is named by the string the kernel's ``get_epilogue_op`` knows,
# with the matching pointwise op for the reference; "identity" is "no epilogue".
_EPILOGUES: dict[str, Callable[[torch.Tensor], torch.Tensor] | None] = {"identity": None, "relu": torch.relu}


def _reference(geometry: ConvGeometry, x: torch.Tensor, dy: torch.Tensor, epilogue: str, out_dtype: torch.dtype) -> torch.Tensor:
    # PyTorch's own convolution weight gradient, evaluated in FP64: the only
    # error left is the kernel's TF32/MMA accumulation and the output rounding.
    dw = torch.nn.grad.conv3d_weight(
        x.double(),
        geometry.w_shape,
        dy.double(),
        stride=geometry.stride,
        padding=geometry.padding,
        dilation=geometry.dilation,
    )
    pointwise = _EPILOGUES[epilogue]
    if pointwise is not None:
        dw = pointwise(dw)
    return dw.to(out_dtype)


def _run(geometry: ConvGeometry, torch_dtype: torch.dtype, epilogue: str) -> None:
    import cutlass
    from cuda.bindings import driver as cuda_driver

    from cudnn.conv.frost.templates.sm100_conv_wgrad import compile as compile_wgrad
    from cudnn.conv.frost.tile_config import ConvTileConfig

    cutlass_dtype = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32}[torch_dtype]

    x = _filled(geometry.x_shape, torch_dtype, seed=1)
    dy = _filled(geometry.dy_shape, torch_dtype, seed=2)
    dw = _channels_last_3d(geometry.w_shape, torch_dtype).fill_(float("nan"))

    # One artifact per (AB dtype, C dtype, tile config, epilogue). The tile is
    # cta_tile_m=128 K-out rows, cta_tile_n=128 C-in channels, a 128-byte
    # reduction stage, and one CTA per cluster.
    #
    # Build after the operands exist: a FROST build retains and then releases the
    # device's primary context, so a build that is the process's first CUDA user
    # would destroy it and the next build would run against a stale handle.
    config = ConvTileConfig(128, 128, 128, 1)
    artifact = compile_wgrad(cutlass_dtype, cutlass_dtype, config, epilogue)

    # Every operand is addressed channel-innermost. A channels-last-3d tensor is
    # already laid out that way; permute only relabels the axes for the kernel.
    x_view, dy_view, dw_view = (t.permute(0, 2, 3, 4, 1) for t in (x, dy, dw))

    # The four geometry triplets are Int32 launch arguments, so the same artifact
    # serves any padding/stride/dilation the im2col tensor map can describe.
    geometry_args = tuple(
        tuple(cutlass.Int32(int(v)) for v in triple) for triple in (geometry.pre_padding, geometry.padded_end, geometry.stride, geometry.dilation)
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    artifact(x_view, dy_view, dw_view, stream, *geometry_args)
    torch.cuda.synchronize()

    assert not torch.isnan(dw).any(), "the kernel must write every dW element"
    want = _reference(geometry, x, dy, epilogue, torch_dtype)
    # The reference is exact FP64, so what remains is the output dtype's rounding
    # (BF16: a fraction of one code) and the kernel's TF32 tensor-core
    # accumulation, which shows up most at small-magnitude outputs. Both
    # tolerances scale with the magnitude of dW.
    scale = max(want.abs().max().item(), 1.0)
    atol, rtol = {torch.bfloat16: (4e-3, 1e-2), torch.float32: (1e-3, 2e-3)}[torch_dtype]
    torch.testing.assert_close(dw, want, atol=atol * scale, rtol=rtol)
    label = "no epilogue" if epilogue == "identity" else f"fused {epilogue}"
    print(f"[04] PASS    dW {str(torch_dtype).removeprefix('torch.'):>6} {label:<11} |dW|max={want.abs().max().item():.4f}")


def main() -> None:
    # The dense FROST kernels need SM100/SM103 and a CuTe DSL new enough for the
    # im2col tensor map. Both are checked before the kernel module (and with it
    # the CuTe DSL) is imported, so an unsupported machine fails with a reason.
    from cudnn.conv.frost.checks import DENSE_CUTEDSL_MIN_VERSION
    from cudnn.frost.buffers import cutedsl_requirement_error

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        raise SystemExit(f"FROST conv wgrad requires SM100 or SM103, found sm_{capability[0]}{capability[1]}")
    error = cutedsl_requirement_error("FROST conv wgrad example", DENSE_CUTEDSL_MIN_VERSION, require_installed=True)
    if error is not None:
        raise SystemExit(error)

    # A padded, undilated 3x3x3 convolution whose reduction is n*z*p*q = 640
    # output pixels; the padded positions are masked by the im2col tensor map.
    geometry = ConvGeometry(n=2, c=128, d=4, h=8, w=10, k=256, t=3, r=3, s=3, pre_padding=(1, 1, 1))
    for torch_dtype in (torch.bfloat16, torch.float32):
        for epilogue in _EPILOGUES:
            _run(geometry, torch_dtype, epilogue)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
