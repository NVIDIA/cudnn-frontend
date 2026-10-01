# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WAN encoder using cuDNN Conv3D post-operation kernels at eligible sites."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from cudnn import (
    causal_conv3d_with_cache_wrapper_sm100,
    conv3d_bias_residual_pad_wrapper_sm100,
    conv3d_raw_wrapper_sm100,
    conv3d_rmsnorm_silu_pad_wrapper_sm100,
    pack_causal_conv3d_weight_sm100,
    pack_conv3d_weight_sm100,
    rmsnorm_silu_pad_wrapper_sm100,
)
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanMidBlock,
    WanResidualBlock,
    WanResidualDownBlock,
)


@torch.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
def _copy_history(destination: torch.Tensor, source: torch.Tensor) -> None:
    """Copy cached history directly into its destination view."""
    destination.copy_(source)


@torch.compiler.disable
def _conv3d_rmsnorm_silu_pad(
    padded_input: torch.Tensor,
    packed_weight: torch.Tensor,
    bias: torch.Tensor,
    gamma: torch.Tensor,
    previous: torch.Tensor | None,
    residual: torch.Tensor | None = None,
    residual_bias: torch.Tensor | None = None,
):
    """Run fused convolution/preparation and fill the caller-owned history regions."""
    history_frames = 0 if previous is None else previous.shape[1]
    result = conv3d_rmsnorm_silu_pad_wrapper_sm100(
        padded_input,
        packed_weight,
        bias,
        gamma,
        history_frames=history_frames,
        residual=residual,
        residual_bias=residual_bias,
    )
    if previous is not None:
        padded, cache = result["padded_output"], result["cache_output"]
        # Pass destination views directly to avoid full-buffer slice updates.
        _copy_history(padded[:, 2 - history_frames : 2, 1:-1, 1:-1, :], previous)
        old_frames = max(cache.shape[1] - (padded.shape[1] - 2), 0)
        if old_frames:
            _copy_history(cache[:, :old_frames], previous[:, -old_frames:])
    return result


@torch.compiler.disable
def _conv3d_bias_residual_pad(
    padded_input: torch.Tensor,
    packed_weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    residual_bias: torch.Tensor | None,
) -> torch.Tensor:
    """Return convolution plus bias/residual with bottom/right spatial padding."""
    return conv3d_bias_residual_pad_wrapper_sm100(
        padded_input,
        packed_weight,
        bias,
        residual,
        residual_bias=residual_bias,
    )["padded_output"]


@torch.compiler.disable
def _conv3d_raw(
    padded_input: torch.Tensor,
    packed_weight: torch.Tensor,
) -> torch.Tensor:
    """Run valid convolution without bias or other post-operations."""
    return conv3d_raw_wrapper_sm100(padded_input, packed_weight)


@torch.compiler.disable
def _causal_conv3d(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    previous: torch.Tensor | None,
):
    """Pack causal input/history, convolve, and return output with the updated cache."""
    return causal_conv3d_with_cache_wrapper_sm100(input, packed_weight, previous=previous)


@torch.compiler.disable
def _rmsnorm_silu_pad(
    input: torch.Tensor,
    gamma: torch.Tensor,
    input_bias: torch.Tensor | None,
    previous: torch.Tensor | None,
    residual: torch.Tensor | None = None,
    residual_bias: torch.Tensor | None = None,
    *,
    save_input: bool = False,
):
    """Normalize and activate input with padding, history, and optional residual fusion."""
    return rmsnorm_silu_pad_wrapper_sm100(
        input,
        gamma,
        input_bias=input_bias,
        previous=previous,
        residual=residual,
        residual_bias=residual_bias,
        save_input=save_input,
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
        x = F.conv2d(
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

    def __init__(self, stage: WanResidualDownBlock) -> None:
        """Reuse the C160/C320 stage modules and prepack their convolution weights."""
        super().__init__()
        if len(stage.resnets) != 2 or stage.downsampler is None:
            raise ValueError("fused WAN stage requires two residual blocks and a downsampler")
        first, second = stage.resnets
        if first.conv1.out_channels not in (160, 320) or second.conv1.out_channels not in (160, 320):
            raise ValueError("fused WAN stage requires C160 or C320 residual blocks")
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
        self.downsampler = stage.downsampler
        self.avg_shortcut = stage.avg_shortcut
        self.register_buffer("first_conv1_weight", pack_conv3d_weight_sm100(first.conv1.weight), persistent=False)
        self.register_buffer("first_conv2_weight", pack_conv3d_weight_sm100(first.conv2.weight), persistent=False)
        self.register_buffer("second_conv1_weight", pack_conv3d_weight_sm100(second.conv1.weight), persistent=False)
        self.register_buffer("second_conv2_weight", pack_conv3d_weight_sm100(second.conv2.weight), persistent=False)

    def forward(self, x: torch.Tensor, feat_cache: list, feat_idx: list[int], input_bias: torch.Tensor | None = None) -> torch.Tensor:
        """Encode a chunk through fused residual blocks and the stage downsampler."""
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
        )
        self._consume_prepared_cache(second_conv2_input, feat_cache, feat_idx)

        padded = _conv3d_bias_residual_pad(
            second_conv2_input["padded_output"],
            self.second_conv2_weight,
            self.second.conv2.bias,
            second_residual,
            None,
        )
        x = self._spatial_downsample(padded, feat_cache, feat_idx)
        return x + self.avg_shortcut(stage_shortcut)


class _SplitResidualStage(_ResidualStage):
    """Use raw Conv3D and standalone post-operation preparation for C640."""

    def __init__(self, stage: WanResidualDownBlock) -> None:
        """Reuse the C640 stage modules and prepack their convolution weights."""
        super().__init__()
        if len(stage.resnets) != 2:
            raise ValueError("split WAN stage requires two residual blocks")
        first, second = stage.resnets
        if first.conv1.out_channels != 640 or second.conv1.out_channels != 640:
            raise ValueError("split WAN stage requires C640 residual outputs")
        for block in (first, second):
            if block.dropout.p != 0.0:
                raise ValueError("split WAN stage requires zero dropout")
            for conv in (block.conv1, block.conv2):
                if tuple(conv._padding) != (1, 1, 1, 1, 2, 0):
                    raise ValueError("split WAN stage requires causal 3x3x3 residual convolutions")
            for norm in (block.norm1, block.norm2):
                if norm.bias != 0.0 or norm.scale != math.sqrt(norm.gamma.numel()):
                    raise ValueError("split WAN stage requires bias-free channel-first RMSNorm")

        self.first = first
        self.second = second
        self.downsampler = stage.downsampler
        self.avg_shortcut = stage.avg_shortcut
        self.register_buffer("first_conv1_weight", pack_conv3d_weight_sm100(first.conv1.weight), persistent=False)
        self.register_buffer("first_conv2_weight", pack_conv3d_weight_sm100(first.conv2.weight), persistent=False)
        self.register_buffer("second_conv1_weight", pack_conv3d_weight_sm100(second.conv1.weight), persistent=False)
        self.register_buffer("second_conv2_weight", pack_conv3d_weight_sm100(second.conv2.weight), persistent=False)

    def forward(self, x: torch.Tensor, feat_cache: list, feat_idx: list[int]) -> torch.Tensor:
        """Encode a C640 chunk using raw convolutions and separate fused post-operations."""
        stage_shortcut = x
        first_residual, first_residual_bias = self._shortcut(self.first, x)
        first_input = self._prepare_first_input(self.first, x, feat_cache, feat_idx)

        first_conv1 = _conv3d_raw(first_input, self.first_conv1_weight)
        first_conv2_input = _rmsnorm_silu_pad(
            first_conv1,
            self.first.norm2.gamma.reshape(-1),
            self.first.conv1.bias,
            feat_cache[feat_idx[0]],
        )
        self._consume_prepared_cache(first_conv2_input, feat_cache, feat_idx)

        first_conv2 = _conv3d_raw(first_conv2_input["padded_output"], self.first_conv2_weight)
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

        second_conv1 = _conv3d_raw(second_conv1_input["padded_output"], self.second_conv1_weight)
        second_conv2_input = _rmsnorm_silu_pad(
            second_conv1,
            self.second.norm2.gamma.reshape(-1),
            self.second.conv1.bias,
            feat_cache[feat_idx[0]],
        )
        self._consume_prepared_cache(second_conv2_input, feat_cache, feat_idx)

        if self.downsampler is not None:
            padded = _conv3d_bias_residual_pad(
                second_conv2_input["padded_output"],
                self.second_conv2_weight,
                self.second.conv2.bias,
                second_residual,
                None,
            )
            x = self._spatial_downsample(padded, feat_cache, feat_idx)
        else:
            second_conv2 = _conv3d_raw(second_conv2_input["padded_output"], self.second_conv2_weight)
            biased = (second_conv2.float() + self.second.conv2.bias.float()).to(torch.bfloat16)
            x = (biased.float() + second_residual.float()).to(torch.bfloat16).permute(0, 4, 1, 2, 3)
        return x + self.avg_shortcut(stage_shortcut)


class _PreparedResidualBlock(torch.nn.Module):
    """Run C640 residual convolutions and defer the final bias/residual sum."""

    def __init__(self, block: WanResidualBlock) -> None:
        """Validate a C640 identity-shortcut block and prepack its convolution weights."""
        super().__init__()
        if not isinstance(block.conv_shortcut, torch.nn.Identity) or block.dropout.p != 0.0:
            raise ValueError("prepared middle block requires an identity shortcut and zero dropout")
        if not isinstance(block.nonlinearity, torch.nn.SiLU):
            raise TypeError("prepared middle block requires SiLU")
        for conv in (block.conv1, block.conv2):
            if (
                conv.in_channels != 640
                or conv.out_channels != 640
                or tuple(conv.kernel_size) != (3, 3, 3)
                or tuple(conv._padding) != (1, 1, 1, 1, 2, 0)
                or tuple(conv.padding) != (0, 0, 0)
                or tuple(conv.stride) != (1, 1, 1)
                or tuple(conv.dilation) != (1, 1, 1)
                or conv.groups != 1
            ):
                raise ValueError("prepared middle block requires causal unit-stride C640 3x3x3 convolutions")
        for norm in (block.norm1, block.norm2):
            if norm.bias != 0.0 or norm.scale != math.sqrt(640):
                raise ValueError("prepared middle block requires bias-free C640 RMSNorm")
        self.block = block
        self.register_buffer("conv1_weight", pack_conv3d_weight_sm100(block.conv1.weight), persistent=False)
        self.register_buffer("conv2_weight", pack_conv3d_weight_sm100(block.conv2.weight), persistent=False)

    def forward(self, x: torch.Tensor, feat_cache: list, feat_idx: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run both convolutions, returning the final raw output, bias, and residual."""
        residual = x.permute(0, 2, 3, 4, 1)
        index = feat_idx[0]
        first = _rmsnorm_silu_pad(residual, self.block.norm1.gamma.reshape(-1), None, feat_cache[index])
        feat_cache[index] = first["cache_output"]
        x = _conv3d_raw(first["padded_output"], self.conv1_weight)
        second = _rmsnorm_silu_pad(x, self.block.norm2.gamma.reshape(-1), self.block.conv1.bias, feat_cache[index + 1])
        feat_cache[index + 1] = second["cache_output"]
        feat_idx[0] += 2
        return _conv3d_raw(second["padded_output"], self.conv2_weight), self.block.conv2.bias, residual


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
            biased = (raw.float() + bias.float()).to(torch.bfloat16)
            x = (biased.float() + residual.float()).to(torch.bfloat16).permute(0, 4, 1, 2, 3)
            if attention is not None:
                x = attention(x)
            raw, bias, residual = resnet(x, feat_cache, feat_idx)
        return raw, bias, residual


class CudnnWanEncoder(torch.nn.Module):
    """Use Conv3D APIs for residual stages, the middle block, and output preparation."""

    def __init__(self, encoder: torch.nn.Module) -> None:
        """Replace supported encoder stages and prepack weights, retaining other layers."""
        super().__init__()
        if len(encoder.down_blocks) != 4 or not all(isinstance(stage, WanResidualDownBlock) for stage in encoder.down_blocks):
            raise ValueError("custom WAN encoder requires the WAN 2.2 four-stage residual architecture")
        self.conv_in = encoder.conv_in
        self.register_buffer(
            "conv_in_weight",
            pack_causal_conv3d_weight_sm100(encoder.conv_in.weight),
            persistent=False,
        )
        self.fused_stages = torch.nn.ModuleList(_FusedResidualStage(stage) for stage in encoder.down_blocks[:2])
        self.split_stages = torch.nn.ModuleList(_SplitResidualStage(stage) for stage in encoder.down_blocks[2:])
        self.mid_block = _PreparedMidBlock(encoder.mid_block)
        self.norm_out = encoder.norm_out
        self.conv_out = encoder.conv_out
        if self.norm_out.bias != 0.0 or self.norm_out.scale != math.sqrt(640) or not isinstance(encoder.nonlinearity, torch.nn.SiLU):
            raise ValueError("prepared output head requires bias-free C640 RMSNorm and SiLU")
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
        result = _causal_conv3d(x, self.conv_in_weight, previous)
        raw = result["output"]
        x = raw.permute(0, 4, 1, 2, 3)
        feat_cache[index] = result["cache_output"]
        feat_idx[0] += 1

        for stage_index, stage in enumerate(self.fused_stages):
            x = stage(x, feat_cache, feat_idx, input_bias=self.conv_in.bias if stage_index == 0 else None)
        for stage in self.split_stages:
            x = stage(x, feat_cache, feat_idx)
        raw, bias, residual = self.mid_block(x, feat_cache, feat_idx)
        index = feat_idx[0]
        prepared = _rmsnorm_silu_pad(raw, self.norm_out.gamma.reshape(-1), bias, feat_cache[index], residual)
        feat_cache[index] = prepared["cache_output"]
        feat_idx[0] += 1
        conv = self.conv_out
        return F.conv3d(prepared["padded_output"].permute(0, 4, 1, 2, 3), conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
