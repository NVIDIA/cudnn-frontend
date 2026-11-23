# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile boundaries for the public cuDNN APIs used by the Wan demo.

Each custom op returns fresh storage. History copying stays with the caller;
strict FP32 convolution stays in Torch, outside the Tensor Core kernel API.
"""

import torch
import torch.nn.functional as F
import cudnn


def custom_convolution(weight):
    """Select supported Tensor Core convolutions; keep wide TF32 sites native."""
    ci, co = weight.shape[1], weight.shape[0]
    pairs = {(96, 96), (96, 192), (192, 192), (192, 384), (384, 384), (160, 160), (160, 320), (320, 320), (320, 640), (640, 640), (512, 256), (256, 256)}
    return (ci, co) in pairs and (
        (weight.dtype == torch.bfloat16 and co != 384) or (weight.dtype == torch.float32 and torch.backends.cudnn.allow_tf32 and co not in (256, 640))
    )


def use_first_plane(weight):
    """Elide zero temporal planes where BF16 accumulation preserves model accuracy."""
    return weight.dtype != torch.bfloat16 or (weight.shape[0] < 512 and weight.shape[:2] != (384, 16))


def spatial_convolution(image, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """Use one-plane native Conv3D for strict FP32 to avoid Conv2D FFT workspaces."""
    if image.dtype == weight.dtype == torch.float32 and not torch.is_autocast_enabled("cuda") and not torch.backends.cudnn.allow_tf32:
        return (
            F.conv3d(
                image.unsqueeze(2).contiguous(),
                weight.unsqueeze(2).contiguous(),
                bias,
                (1, *stride),
                (0, *padding),
                (1, *dilation),
                groups,
            )
            .squeeze(2)
            .contiguous(memory_format=torch.channels_last)
        )
    return F.conv2d(image, weight, bias, stride, padding, dilation, groups)


class _SpatialConv2d(torch.nn.Module):
    """Retain the demo's spatial layers with the shared native arithmetic policy."""

    def __init__(self, conv):
        super().__init__()
        self.weight, self.bias = conv.weight, conv.bias
        self.stride, self.padding, self.dilation, self.groups = conv.stride, conv.padding, conv.dilation, conv.groups

    def forward(self, x):
        return spatial_convolution(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def prepare_spatial_convolutions(module):
    """Apply the same policy to attention projections and spatial resampling."""
    for name, child in module.named_children():
        if type(child) is torch.nn.Conv2d:
            setattr(module, name, _SpatialConv2d(child))
        else:
            prepare_spatial_convolutions(child)


def first_plane(weight, first):
    """Select the measured native first-frame shortcut for residual convolutions."""
    if first and use_first_plane(weight) and (weight.dtype == torch.float32 or weight.shape[0] in (96, 192, 256)):
        return weight[:, :, 2].contiguous(memory_format=torch.channels_last)
    return None


def native_input(conv, x, previous):
    if x.shape[2] == 1 and previous is None and use_first_plane(conv.weight):
        return spatial_convolution(x[:, :, 0], conv.weight[:, :, 2].contiguous(memory_format=torch.channels_last), conv.bias, padding=(1, 1)).unsqueeze(2)
    return conv(x, previous)


def native_output(conv, padded, first):
    if first and use_first_plane(conv.weight):
        return spatial_convolution(padded[:, 2].permute(0, 3, 1, 2), conv.weight[:, :, 2].contiguous(memory_format=torch.channels_last), conv.bias).unsqueeze(2)
    return F.conv3d(padded.permute(0, 4, 1, 2, 3), conv.weight, conv.bias)


def pack(weight):
    if not custom_convolution(weight):
        return weight
    return cudnn.pack_conv3d_weight_sm100(weight, allow_tf32=weight.dtype == torch.float32)


@torch.library.custom_op("wan_demo::raw", mutates_args=())
def _raw(x: torch.Tensor, weight: torch.Tensor, tf32: bool) -> torch.Tensor:
    return cudnn.conv3d_raw_wrapper_sm100(x, weight, allow_tf32=tf32)


@_raw.register_fake
def _raw_fake(x, weight, tf32):
    n, t, h, w, _ = x.shape
    return x.new_empty((n, t - 2, h - 2, w - 2, weight.shape[0]))


def convolve(x, weight, custom, first_frame=False, first_weight=None):
    if first_weight is not None and use_first_plane(weight):
        return spatial_convolution(x[:, 2].permute(0, 3, 1, 2), first_weight).permute(0, 2, 3, 1).unsqueeze(1)
    if first_frame and not custom and use_first_plane(weight):
        return spatial_convolution(x[:, 2].permute(0, 3, 1, 2), weight[:, :, 2].contiguous(memory_format=torch.channels_last)).permute(0, 2, 3, 1).unsqueeze(1)
    if custom:
        return _raw(x, weight, x.dtype == torch.float32)
    return F.conv3d(x.permute(0, 4, 1, 2, 3), weight).permute(0, 2, 3, 4, 1)


@torch.library.custom_op(
    "wan_demo::norm",
    mutates_args=(),
    schema="(Tensor x, Tensor gamma, Tensor? bias, Tensor? previous, Tensor? residual, Tensor? residual_bias, bool save_input) -> (Tensor, Tensor, Tensor?)",
)
def _norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    previous: torch.Tensor | None,
    residual: torch.Tensor | None,
    residual_bias: torch.Tensor | None,
    save_input: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    r = cudnn.rmsnorm_silu_pad_wrapper_sm100(
        x, gamma, input_bias=bias, previous=previous, residual=residual, residual_bias=residual_bias, save_input=save_input
    )
    return r["padded_output"], r["cache_output"], r["residual_output"]


@_norm.register_fake
def _norm_fake(x, gamma, bias, previous, residual, residual_bias, save_input):
    n, t, h, w, c = x.shape
    history = 0 if previous is None else previous.shape[1]
    return (
        x.new_empty((n, t + 2, h + 2, w + 2, c)),
        x.new_empty((n, min(2, t + history), h, w, c)),
        torch.empty_like(x) if residual is not None or save_input else None,
    )


def _result(values):
    return dict(zip(("padded_output", "cache_output", "residual_output"), values))


def norm(x, gamma, input_bias, previous, residual=None, residual_bias=None, *, save_input=False):
    return _result(_norm(x.contiguous(), gamma, input_bias, previous, residual, residual_bias, save_input))


@torch.library.custom_op(
    "wan_demo::conv_norm",
    mutates_args=(),
    schema="(Tensor x, Tensor weight, Tensor bias, Tensor gamma, Tensor? previous, Tensor? residual, Tensor? residual_bias, bool tf32) -> (Tensor, Tensor, Tensor?)",
)
def _conv_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    gamma: torch.Tensor,
    previous: torch.Tensor | None,
    residual: torch.Tensor | None,
    residual_bias: torch.Tensor | None,
    tf32: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    history = 0 if previous is None else previous.shape[1]
    r = cudnn.conv3d_rmsnorm_silu_pad_wrapper_sm100(
        x, weight, bias, gamma, history_frames=history, residual=residual, residual_bias=residual_bias, allow_tf32=tf32
    )
    padded, cache = r["padded_output"], r["cache_output"]
    if previous is not None:
        padded[:, 2 - history : 2, 1:-1, 1:-1].copy_(previous)
        old = max(cache.shape[1] - (padded.shape[1] - 2), 0)
        if old:
            cache[:, :old].copy_(previous[:, -old:])
    return padded, cache, r["residual_output"]


@_conv_norm.register_fake
def _conv_norm_fake(x, weight, bias, gamma, previous, residual, residual_bias, tf32):
    y = _raw_fake(x, weight, tf32)
    return _norm_fake(y, gamma, bias, previous, residual, residual_bias, False)


def conv_norm(x, weight, bias, gamma, previous, residual=None, residual_bias=None, *, first_weight=None):
    if first_weight is not None:
        return norm(convolve(x, weight, False, first_weight=first_weight), gamma, bias, previous, residual, residual_bias)
    # Packed weights are OTRSC; native weights are OI3x3x3.
    custom = weight.shape[1] == 3
    if custom and not (x.dtype == torch.float32 and (weight.shape[0] == 320 or (weight.shape[0] == 160 and residual is not None))):
        return _result(_conv_norm(x, weight, bias, gamma, previous, residual, residual_bias, x.dtype == torch.float32))
    return norm(convolve(x, weight, custom), gamma, bias, previous, residual, residual_bias)


@torch.library.custom_op("wan_demo::conv_spatial", mutates_args=())
def _conv_spatial(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, residual_bias: torch.Tensor | None, tf32: bool
) -> torch.Tensor:
    return cudnn.conv3d_bias_residual_pad_wrapper_sm100(x, weight, bias, residual, residual_bias=residual_bias, allow_tf32=tf32)["padded_output"]


@_conv_spatial.register_fake
def _conv_spatial_fake(x, weight, bias, residual, residual_bias, tf32):
    n, t, h, w, _ = x.shape
    return x.new_empty((n, t - 2, h - 1, w - 1, weight.shape[0]))


def add_residual(raw, bias, residual, residual_bias=None):
    value = (raw.float() + bias.float()).to(raw.dtype)
    if residual_bias is not None:
        residual = (residual.float() + residual_bias.float()).to(raw.dtype)
    return (value.float() + residual.float()).to(raw.dtype)


def conv_spatial(x, weight, bias, residual, residual_bias, *, first_weight=None):
    if first_weight is not None:
        return F.pad(add_residual(convolve(x, weight, False, first_weight=first_weight), bias, residual, residual_bias), (0, 0, 0, 1, 0, 1))
    if weight.shape[1] == 3:
        return _conv_spatial(x, weight, bias, residual, residual_bias, x.dtype == torch.float32)
    return F.pad(add_residual(convolve(x, weight, False), bias, residual, residual_bias), (0, 0, 0, 1, 0, 1))


@torch.library.custom_op("wan_demo::input_conv", mutates_args=())
def _input_conv(x: torch.Tensor, weight: torch.Tensor, previous: torch.Tensor | None, tf32: bool) -> tuple[torch.Tensor, torch.Tensor]:
    r = cudnn.causal_conv3d_with_cache_wrapper_sm100(x, weight, previous, allow_tf32=tf32)
    return r["output"], r["cache_output"]


@_input_conv.register_fake
def _input_conv_fake(x, weight, previous, tf32):
    n, c, t, h, w = x.shape
    history = 0 if previous is None else previous.shape[1]
    return x.new_empty((n, t, h, w, weight.shape[0])), x.new_empty((n, min(2, t + history), h, w, c))


def input_conv(x, weight, previous):
    output, cache = _input_conv(x, weight, previous, x.dtype == torch.float32)
    return {"output": output, "cache_output": cache}
