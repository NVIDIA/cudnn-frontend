# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WAN encoder using cuDNN Conv3D post-operation kernels at eligible sites."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from cudnn import pack_causal_conv3d_weight_sm100
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanMidBlock,
    WanResample,
    WanResidualBlock,
)


from wan_ops import (
    convolve as _convolve,
    conv_norm as _conv3d_rmsnorm_silu_pad,
    conv_spatial as _conv3d_bias_residual_pad,
    input_conv as _causal_conv3d,
    norm as _rmsnorm_silu_pad,
    pack as pack_conv3d_weight_sm100,
    custom_convolution,
    first_plane,
    native_input,
    native_output,
    spatial_convolution,
    prepare_spatial_convolutions,
)


class _ResidualStage(torch.nn.Module):
    """Shared shortcut, cache, and downsampling operations for residual stages."""

    @staticmethod
    def _shortcut(block, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the NTHWC shortcut and any deferred projection bias."""
        if isinstance(block.conv_shortcut, torch.nn.Identity):
            return x.permute(0, 2, 3, 4, 1), None
        conv = block.conv_shortcut
        output = F.conv3d(x, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)
        return output.permute(0, 2, 3, 4, 1), conv.bias

    @staticmethod
    def _prepare_first_input(block, x: torch.Tensor, feat_cache: list, feat_idx: list[int]) -> torch.Tensor:
        """Normalize and pad a block's input while updating its convolution cache."""
        index = feat_idx[0]
        previous = feat_cache[index]
        if previous is not None and not isinstance(previous, torch.Tensor):
            raise TypeError("residual convolution cache must be a tensor or None")
        prepared = _rmsnorm_silu_pad(
            x.permute(0, 2, 3, 4, 1),
            block.norm1.gamma.reshape(-1),
            None,
            previous,
        )
        feat_cache[index] = prepared["cache_output"]
        feat_idx[0] += 1
        return prepared["padded_output"]

    @staticmethod
    def _consume_prepared_cache(prepared, feat_cache: list, feat_idx: list[int]) -> None:
        """Save prepared history and advance the feature-cache index."""
        feat_cache[feat_idx[0]] = prepared["cache_output"]
        feat_idx[0] += 1

    def _spatial_downsample(self, padded: torch.Tensor, feat_cache: list, feat_idx: list[int]) -> torch.Tensor:
        """Apply spatial downsampling and any cached temporal downsampling."""
        n, frames, height, width, channels = padded.shape
        conv = self.downsampler.resample[1]
        x = spatial_convolution(
            padded.reshape(n * frames, height, width, channels).permute(0, 3, 1, 2),
            conv.weight,
            conv.bias,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
        )
        x = x.view(n, frames, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
        if self.downsampler.mode == "downsample3d":
            index = feat_idx[0]
            previous = feat_cache[index]
            if previous is None:
                feat_cache[index] = x.clone()
            else:
                if not isinstance(previous, torch.Tensor):
                    raise TypeError("temporal downsample cache must be a tensor or None")
                cache = x[:, :, -1:].clone()
                x = self.downsampler.time_conv(torch.cat((previous[:, :, -1:], x), dim=2))
                feat_cache[index] = cache
            feat_idx[0] += 1
        return x


class _FusedResidualStage(_ResidualStage):
    """Fuse across both residual blocks and into the following downsample."""

    def __init__(self, resnets, downsampler, avg_shortcut=None) -> None:
        """Reuse the C96/C192/C160/C320 stage modules and prepack their convolution weights."""
        super().__init__()
        if len(resnets) != 2 or downsampler is None:
            raise ValueError("fused WAN stage requires two residual blocks and a downsampler")
        first, second = resnets
        if first.conv1.out_channels not in (96, 192, 160, 320) or second.conv1.out_channels not in (96, 192, 160, 320):
            raise ValueError("fused WAN stage requires C96/C192/C160/C320 residual blocks")
        for block in (first, second):
            if block.dropout.p != 0.0:
                raise ValueError("fused WAN stage requires zero dropout")
            for conv in (block.conv1, block.conv2):
                if tuple(conv._padding) != (1, 1, 1, 1, 2, 0):
                    raise ValueError("fused WAN stage requires causal 3x3x3 residual convolutions")
            for norm in (block.norm1, block.norm2):
                if norm.bias != 0.0 or norm.scale != math.sqrt(norm.gamma.numel()):
                    raise ValueError("fused WAN stage requires bias-free channel-first RMSNorm")

        self.first = first
        self.second = second
        self.downsampler = downsampler
        self.avg_shortcut = avg_shortcut
        self.register_buffer("first_conv1_weight", pack_conv3d_weight_sm100(first.conv1.weight), persistent=False)
        self.register_buffer("first_conv2_weight", pack_conv3d_weight_sm100(first.conv2.weight), persistent=False)
        self.register_buffer("second_conv1_weight", pack_conv3d_weight_sm100(second.conv1.weight), persistent=False)
        self.register_buffer("second_conv2_weight", pack_conv3d_weight_sm100(second.conv2.weight), persistent=False)

    def forward(self, x: torch.Tensor, feat_cache: list, feat_idx: list[int], input_bias: torch.Tensor | None = None) -> torch.Tensor:
        """Encode a chunk through fused residual blocks and the stage downsampler."""
        first_frame = x.shape[2] == 1 and feat_cache[feat_idx[0]] is None
        if input_bias is not None:
            prepared = _rmsnorm_silu_pad(
                x.permute(0, 2, 3, 4, 1),
                self.first.norm1.gamma.reshape(-1),
                input_bias,
                feat_cache[feat_idx[0]],
                save_input=True,
            )
            self._consume_prepared_cache(prepared, feat_cache, feat_idx)
            x = prepared["residual_output"].permute(0, 4, 1, 2, 3)
            first_input = prepared["padded_output"]
        else:
            first_input = self._prepare_first_input(self.first, x, feat_cache, feat_idx)
        stage_shortcut = x
        first_residual, first_residual_bias = self._shortcut(self.first, x)

        first_conv2_input = _conv3d_rmsnorm_silu_pad(
            first_input,
            self.first_conv1_weight,
            self.first.conv1.bias,
            self.first.norm2.gamma.reshape(-1),
            feat_cache[feat_idx[0]],
            first_weight=first_plane(self.first.conv1.weight, first_frame),
        )
        self._consume_prepared_cache(first_conv2_input, feat_cache, feat_idx)

        second_conv1_input = _conv3d_rmsnorm_silu_pad(
            first_conv2_input["padded_output"],
            self.first_conv2_weight,
            self.first.conv2.bias,
            self.second.norm1.gamma.reshape(-1),
            feat_cache[feat_idx[0]],
            first_residual,
            first_residual_bias,
            first_weight=first_plane(self.first.conv2.weight, first_frame),
        )
        second_residual = second_conv1_input["residual_output"]
        if second_residual is None:
            raise RuntimeError("inter-block fusion did not return the residual output")
        self._consume_prepared_cache(second_conv1_input, feat_cache, feat_idx)

        second_conv2_input = _conv3d_rmsnorm_silu_pad(
            second_conv1_input["padded_output"],
            self.second_conv1_weight,
            self.second.conv1.bias,
            self.second.norm2.gamma.reshape(-1),
            feat_cache[feat_idx[0]],
            first_weight=first_plane(self.second.conv1.weight, first_frame),
        )
        self._consume_prepared_cache(second_conv2_input, feat_cache, feat_idx)

        padded = _conv3d_bias_residual_pad(
            second_conv2_input["padded_output"],
            self.second_conv2_weight,
            self.second.conv2.bias,
            second_residual,
            None,
            first_weight=first_plane(self.second.conv2.weight, first_frame),
        )
        x = self._spatial_downsample(padded, feat_cache, feat_idx)
        return x if self.avg_shortcut is None else x + self.avg_shortcut(stage_shortcut)


class _SplitResidualStage(_ResidualStage):
    """Use separate convolution and post-operation preparation for C384/C640."""

    def __init__(self, resnets, downsampler, avg_shortcut=None) -> None:
        """Keep C384 convolutions in Torch; prepack C640 weights for the library."""
        super().__init__()
        if len(resnets) != 2:
            raise ValueError("split WAN stage requires two residual blocks")
        first, second = resnets
        if first.conv1.out_channels not in (96, 192, 256, 384, 512, 640, 1024) or second.conv1.out_channels != first.conv1.out_channels:
            raise ValueError("split WAN stage requires C384 or C640 residual outputs")
        for block in (first, second):
            if block.dropout.p != 0.0:
                raise ValueError("split WAN stage requires zero dropout")
            for conv in (block.conv1, block.conv2):
                if tuple(conv._padding) != (1, 1, 1, 1, 2, 0):
                    raise ValueError("split WAN stage requires causal 3x3x3 residual convolutions")
            for norm in (block.norm1, block.norm2):
                if norm.bias != 0.0 or norm.scale != math.sqrt(norm.gamma.numel()):
                    raise ValueError("split WAN stage requires bias-free channel-first RMSNorm")

        self.custom_conv = custom_convolution(first.conv1.weight)
        self.first = first
        self.second = second
        self.downsampler = downsampler
        self.avg_shortcut = avg_shortcut
        self.register_buffer("first_conv1_weight", pack_conv3d_weight_sm100(first.conv1.weight) if self.custom_conv else first.conv1.weight, persistent=False)
        self.register_buffer("first_conv2_weight", pack_conv3d_weight_sm100(first.conv2.weight) if self.custom_conv else first.conv2.weight, persistent=False)
        self.register_buffer(
            "second_conv1_weight", pack_conv3d_weight_sm100(second.conv1.weight) if self.custom_conv else second.conv1.weight, persistent=False
        )
        self.register_buffer(
            "second_conv2_weight", pack_conv3d_weight_sm100(second.conv2.weight) if self.custom_conv else second.conv2.weight, persistent=False
        )

    def forward(self, x: torch.Tensor, feat_cache: list, feat_idx: list[int]) -> torch.Tensor:
        """Encode a wide-channel chunk using raw convolutions and separate fused post-operations."""
        first = x.shape[2] == 1 and feat_cache[feat_idx[0]] is None
        stage_shortcut = x
        first_residual, first_residual_bias = self._shortcut(self.first, x)
        first_input = self._prepare_first_input(self.first, x, feat_cache, feat_idx)

        first_conv1 = _convolve(first_input, self.first_conv1_weight, self.custom_conv, first, first_plane(self.first.conv1.weight, first))
        first_conv2_input = _rmsnorm_silu_pad(
            first_conv1,
            self.first.norm2.gamma.reshape(-1),
            self.first.conv1.bias,
            feat_cache[feat_idx[0]],
        )
        self._consume_prepared_cache(first_conv2_input, feat_cache, feat_idx)

        first_conv2 = _convolve(
            first_conv2_input["padded_output"], self.first_conv2_weight, self.custom_conv, first, first_plane(self.first.conv2.weight, first)
        )
        second_conv1_input = _rmsnorm_silu_pad(
            first_conv2,
            self.second.norm1.gamma.reshape(-1),
            self.first.conv2.bias,
            feat_cache[feat_idx[0]],
            first_residual,
            first_residual_bias,
        )
        second_residual = second_conv1_input["residual_output"]
        if second_residual is None:
            raise RuntimeError("inter-block preparation did not return the residual output")
        self._consume_prepared_cache(second_conv1_input, feat_cache, feat_idx)

        second_conv1 = _convolve(
            second_conv1_input["padded_output"], self.second_conv1_weight, self.custom_conv, first, first_plane(self.second.conv1.weight, first)
        )
        second_conv2_input = _rmsnorm_silu_pad(
            second_conv1,
            self.second.norm2.gamma.reshape(-1),
            self.second.conv1.bias,
            feat_cache[feat_idx[0]],
        )
        self._consume_prepared_cache(second_conv2_input, feat_cache, feat_idx)

        if self.downsampler is not None and self.custom_conv:
            padded = _conv3d_bias_residual_pad(
                second_conv2_input["padded_output"],
                self.second_conv2_weight,
                self.second.conv2.bias,
                second_residual,
                None,
                first_weight=first_plane(self.second.conv2.weight, first),
            )
            x = self._spatial_downsample(padded, feat_cache, feat_idx)
        else:
            second_conv2 = _convolve(
                second_conv2_input["padded_output"], self.second_conv2_weight, self.custom_conv, first, first_plane(self.second.conv2.weight, first)
            )
            biased = (second_conv2.float() + self.second.conv2.bias.float()).to(second_conv2.dtype)
            x = (biased.float() + second_residual.float()).to(second_conv2.dtype).permute(0, 4, 1, 2, 3)
            if self.downsampler is not None:
                x = self.downsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)
        return x if self.avg_shortcut is None else x + self.avg_shortcut(stage_shortcut)


class _PreparedResidualBlock(torch.nn.Module):
    """Run residual convolutions and defer the final bias/residual sum."""

    def __init__(self, block: WanResidualBlock) -> None:
        """Prepare an identity-shortcut block using the matching convolution backend."""
        super().__init__()
        if not isinstance(block.conv_shortcut, torch.nn.Identity) or block.dropout.p != 0.0:
            raise ValueError("prepared middle block requires an identity shortcut and zero dropout")
        if not isinstance(block.nonlinearity, torch.nn.SiLU):
            raise TypeError("prepared middle block requires SiLU")
        channels = block.conv1.out_channels
        self.custom_conv = custom_convolution(block.conv1.weight)
        for conv in (block.conv1, block.conv2):
            if (
                channels not in (96, 192, 256, 384, 512, 640, 1024)
                or conv.in_channels != channels
                or conv.out_channels != channels
                or tuple(conv.kernel_size) != (3, 3, 3)
                or tuple(conv._padding) != (1, 1, 1, 1, 2, 0)
                or tuple(conv.padding) != (0, 0, 0)
                or tuple(conv.stride) != (1, 1, 1)
                or tuple(conv.dilation) != (1, 1, 1)
                or conv.groups != 1
            ):
                raise ValueError("prepared middle block requires supported causal unit-stride 3x3x3 convolutions")
        for norm in (block.norm1, block.norm2):
            if norm.bias != 0.0 or norm.scale != math.sqrt(channels):
                raise ValueError("prepared middle block requires bias-free channel RMSNorm")
        self.block = block
        self.register_buffer("conv1_weight", pack_conv3d_weight_sm100(block.conv1.weight) if self.custom_conv else block.conv1.weight, persistent=False)
        self.register_buffer("conv2_weight", pack_conv3d_weight_sm100(block.conv2.weight) if self.custom_conv else block.conv2.weight, persistent=False)

    def forward(self, x: torch.Tensor, feat_cache: list, feat_idx: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run both convolutions, returning the final raw output, bias, and residual."""
        residual = x.permute(0, 2, 3, 4, 1)
        index = feat_idx[0]
        first_frame = x.shape[2] == 1 and feat_cache[index] is None
        first = _rmsnorm_silu_pad(residual, self.block.norm1.gamma.reshape(-1), None, feat_cache[index])
        feat_cache[index] = first["cache_output"]
        x = _convolve(first["padded_output"], self.conv1_weight, self.custom_conv, first_frame, first_plane(self.block.conv1.weight, first_frame))
        second = _rmsnorm_silu_pad(x, self.block.norm2.gamma.reshape(-1), self.block.conv1.bias, feat_cache[index + 1])
        feat_cache[index + 1] = second["cache_output"]
        feat_idx[0] += 2
        return (
            _convolve(second["padded_output"], self.conv2_weight, self.custom_conv, first_frame, first_plane(self.block.conv2.weight, first_frame)),
            self.block.conv2.bias,
            residual,
        )


class _PreparedMidBlock(torch.nn.Module):
    """Reuse Diffusers attention between prepared residual blocks."""

    def __init__(self, block: WanMidBlock) -> None:
        """Wrap the middle residual blocks while retaining their attention modules."""
        super().__init__()
        if not block.resnets or len(block.attentions) != len(block.resnets) - 1:
            raise ValueError("prepared middle block requires alternating residual and attention blocks")
        self.resnets = torch.nn.ModuleList(_PreparedResidualBlock(resnet) for resnet in block.resnets)
        self.attentions = block.attentions

    def forward(self, x: torch.Tensor, feat_cache: list, feat_idx: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply middle-block attention while deferring the final bias/residual sum."""
        raw, bias, residual = self.resnets[0](x, feat_cache, feat_idx)
        for attention, resnet in zip(self.attentions, self.resnets[1:]):
            # Preserve both BF16 rounding points before the attention block.
            biased = (raw.float() + bias.float()).to(raw.dtype)
            x = (biased.float() + residual.float()).to(raw.dtype).permute(0, 4, 1, 2, 3)
            if attention is not None:
                x = attention(x)
            raw, bias, residual = resnet(x, feat_cache, feat_idx)
        return raw, bias, residual


class CudnnWanEncoder(torch.nn.Module):
    """Use Conv3D APIs for residual stages, the middle block, and output preparation."""

    def __init__(self, encoder: torch.nn.Module, *, is_residual: bool = True) -> None:
        """Replace supported encoder stages and prepack weights, retaining other layers."""
        super().__init__()
        prepare_spatial_convolutions(encoder)
        self.is_residual = is_residual
        self.custom_input = is_residual and (encoder.conv_in.weight.dtype == torch.bfloat16 or torch.backends.cudnn.allow_tf32)
        self.conv_in = encoder.conv_in
        stages = []
        if is_residual:
            if self.custom_input:
                self.register_buffer(
                    "conv_in_weight",
                    pack_causal_conv3d_weight_sm100(encoder.conv_in.weight, allow_tf32=encoder.conv_in.weight.dtype == torch.float32),
                    persistent=False,
                )
            for stage in encoder.down_blocks:
                cls = _FusedResidualStage if stage.resnets[0].conv1.out_channels in (160, 320) else _SplitResidualStage
                stages.append(cls(stage.resnets, stage.downsampler, stage.avg_shortcut))
        else:
            # Wan 2.1 has flat residual/downsample blocks and no outer shortcut.
            resnets = []
            for block in encoder.down_blocks:
                if isinstance(block, WanResidualBlock):
                    resnets.append(block)
                elif isinstance(block, WanResample):
                    cls = _FusedResidualStage if resnets[0].conv1.out_channels in (96, 192) else _SplitResidualStage
                    stages.append(cls(resnets, block))
                    resnets = []
                else:
                    raise TypeError("expected Wan 2.1 residual blocks and downsamplers")
            stages.append(_SplitResidualStage(resnets, None))
        self.stages = torch.nn.ModuleList(stages)
        self.mid_block = _PreparedMidBlock(encoder.mid_block)
        self.norm_out = encoder.norm_out
        self.conv_out = encoder.conv_out
        if self.norm_out.bias != 0.0 or self.norm_out.scale != math.sqrt(self.norm_out.gamma.numel()) or not isinstance(encoder.nonlinearity, torch.nn.SiLU):
            raise ValueError("prepared output head requires bias-free channel RMSNorm and SiLU")
        if (
            tuple(self.conv_out.kernel_size) != (3, 3, 3)
            or tuple(self.conv_out._padding) != (1, 1, 1, 1, 2, 0)
            or tuple(self.conv_out.padding) != (0, 0, 0)
            or tuple(self.conv_out.stride) != (1, 1, 1)
            or tuple(self.conv_out.dilation) != (1, 1, 1)
            or self.conv_out.groups != 1
        ):
            raise ValueError("prepared output head requires a causal unit-stride 3x3x3 convolution")

    def forward(self, x: torch.Tensor, feat_cache=None, feat_idx=None) -> torch.Tensor:
        """Encode one video chunk and update its streaming feature caches."""
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is None:
            raise ValueError("custom WAN encoder requires a feature cache")

        index = feat_idx[0]
        previous = feat_cache[index]
        if previous is not None and not isinstance(previous, torch.Tensor):
            raise TypeError("input convolution cache must be a tensor or None")
        if self.custom_input:
            result = _causal_conv3d(x, self.conv_in_weight, previous)
            x = result["output"].permute(0, 4, 1, 2, 3)
            feat_cache[index] = result["cache_output"]
            input_bias = self.conv_in.bias
        else:
            # The C3 input convolution stays in Torch with Diffusers cache semantics.
            cache = x[:, :, -2:].clone()
            if cache.shape[2] < 2 and previous is not None:
                cache = torch.cat((previous[:, :, -1:], cache), dim=2)
            x = native_input(self.conv_in, x, previous)
            feat_cache[index] = cache
            input_bias = None
        feat_idx[0] += 1

        x = self.stages[0](x, feat_cache, feat_idx, input_bias=input_bias)
        for stage in self.stages[1:]:
            x = stage(x, feat_cache, feat_idx)
        raw, bias, residual = self.mid_block(x, feat_cache, feat_idx)
        index = feat_idx[0]
        first = raw.shape[1] == 1 and feat_cache[index] is None
        prepared = _rmsnorm_silu_pad(raw, self.norm_out.gamma.reshape(-1), bias, feat_cache[index], residual)
        feat_cache[index] = prepared["cache_output"]
        feat_idx[0] += 1
        return native_output(self.conv_out, prepared["padded_output"], first)
