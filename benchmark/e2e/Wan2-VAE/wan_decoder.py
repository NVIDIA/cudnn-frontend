# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan decoder demo with stage-level cuDNN convolution/post-operation fusion."""

import torch
import torch.nn.functional as F

from wan_encoder import _PreparedMidBlock
from wan_ops import prepare_spatial_convolutions, add_residual, convolve, conv_norm, custom_convolution, native_input, native_output, norm, pack


class _UpStage(torch.nn.Module):
    """Carry prepared convolution input across a decoder stage's residual blocks."""

    def __init__(self, stage, is_residual):
        super().__init__()
        self.resnets = stage.resnets
        self.upsampler = stage.upsampler if is_residual else (None if stage.upsamplers is None else stage.upsamplers[0])
        self.avg_shortcut = stage.avg_shortcut if is_residual else None
        self.custom = []
        for i, block in enumerate(self.resnets):
            custom = custom_convolution(block.conv1.weight)
            self.custom.append(custom)
            for name in ("conv1", "conv2"):
                weight = getattr(block, name).weight
                self.register_buffer(f"weight_{i}_{name}", pack(weight), persistent=False)
                self.register_buffer(f"first_{i}_{name}", weight[:, :, 2].contiguous(memory_format=torch.channels_last), persistent=False)

    def forward(self, x, feat_cache, feat_idx, first_chunk=False):
        source = x
        x = x.permute(0, 2, 3, 4, 1)
        prepared = None
        prepared_first = False
        for i, block in enumerate(self.resnets):
            slot = feat_idx[0]
            first = prepared_first if prepared is not None else x.shape[1] == 1 and feat_cache[slot] is None
            first2 = x.shape[1] == 1 and feat_cache[slot + 1] is None
            use_first = first and (x.dtype == torch.float32 or block.conv1.out_channels in (96, 192, 256))
            use_first2 = first2 and (x.dtype == torch.float32 or block.conv2.out_channels in (96, 192, 256))
            if prepared is None:
                prepared = norm(x, block.norm1.gamma.reshape(-1), None, feat_cache[slot])
            feat_cache[slot] = prepared["cache_output"]
            residual, residual_bias = x, None
            if not isinstance(block.conv_shortcut, torch.nn.Identity):
                residual = F.conv3d(x.permute(0, 4, 1, 2, 3), block.conv_shortcut.weight).permute(0, 2, 3, 4, 1)
                residual_bias = block.conv_shortcut.bias
            w1, w2 = getattr(self, f"weight_{i}_conv1"), getattr(self, f"weight_{i}_conv2")
            fused = self.custom[i] and block.conv1.out_channels in (96, 160, 192, 256, 320)
            if fused and not use_first:
                prepared = conv_norm(prepared["padded_output"], w1, block.conv1.bias, block.norm2.gamma.reshape(-1), feat_cache[slot + 1])
            else:
                raw = convolve(prepared["padded_output"], w1, self.custom[i], first, getattr(self, f"first_{i}_conv1") if use_first else None)
                prepared = norm(raw, block.norm2.gamma.reshape(-1), block.conv1.bias, feat_cache[slot + 1])
            feat_cache[slot + 1] = prepared["cache_output"]
            feat_idx[0] += 2
            gamma = self.resnets[i + 1].norm1.gamma.reshape(-1) if i + 1 < len(self.resnets) else None
            if gamma is not None:
                previous = feat_cache[feat_idx[0]]
                prepared_first = x.shape[1] == 1 and previous is None
                if fused and not use_first2 and x.dtype != torch.float32:
                    prepared = conv_norm(prepared["padded_output"], w2, block.conv2.bias, gamma, previous, residual, residual_bias)
                else:
                    raw = convolve(prepared["padded_output"], w2, self.custom[i], first2, getattr(self, f"first_{i}_conv2") if use_first2 else None)
                    prepared = norm(raw, gamma, block.conv2.bias, previous, residual, residual_bias)
                x = prepared["residual_output"]
            else:
                raw = convolve(prepared["padded_output"], w2, self.custom[i], first2, getattr(self, f"first_{i}_conv2") if use_first2 else None)
                x = add_residual(raw, block.conv2.bias, residual, residual_bias).permute(0, 4, 1, 2, 3)
        if self.upsampler is not None:
            x = self.upsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(source, first_chunk=first_chunk)
        return x


class CudnnWanDecoder(torch.nn.Module):
    """Retain Diffusers temporal orchestration and replace residual preparation."""

    def __init__(self, decoder, *, is_residual):
        super().__init__()
        prepare_spatial_convolutions(decoder)
        self.conv_in = decoder.conv_in
        self.mid_block = _PreparedMidBlock(decoder.mid_block)
        self.stages = torch.nn.ModuleList(_UpStage(stage, is_residual) for stage in decoder.up_blocks)
        self.norm_out = decoder.norm_out
        self.conv_out = decoder.conv_out

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        if feat_idx is None:
            feat_idx = [0]
        slot = feat_idx[0]
        previous = feat_cache[slot]
        cache = x[:, :, -2:].clone()
        if cache.shape[2] < 2 and previous is not None:
            cache = torch.cat((previous[:, :, -1:], cache), dim=2)
        x = native_input(self.conv_in, x, previous)
        feat_cache[slot] = cache
        feat_idx[0] += 1
        raw, bias, residual = self.mid_block(x, feat_cache, feat_idx)
        x = add_residual(raw, bias, residual).permute(0, 4, 1, 2, 3)
        for stage in self.stages:
            x = stage(x, feat_cache, feat_idx, first_chunk)
        slot = feat_idx[0]
        first = x.shape[2] == 1 and feat_cache[slot] is None
        prepared = norm(x.permute(0, 2, 3, 4, 1), self.norm_out.gamma.reshape(-1), None, feat_cache[slot])
        feat_cache[slot] = prepared["cache_output"]
        feat_idx[0] += 1
        padded = prepared["padded_output"]
        return native_output(self.conv_out, padded, first)
