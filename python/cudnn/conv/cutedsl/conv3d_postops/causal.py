# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""C12 input/history packing and the specialized causal Conv3D."""

import math
from dataclasses import dataclass

import cutlass
from cuda.bindings import driver as cuda_driver
from cutlass import cute
from cutlass.experimental import cuda

from cudnn._cutlass_helpers.static_persistent_tile_scheduler import (
    PersistentTileSchedulerParams,
    StaticPersistentTileScheduler,
)

from .kernel import _contiguous_tma_layout, _conv3d_postops_kernel


@cute.kernel
def _pack_c12_input_kernel(
    x: cute.Tensor,
    previous: cute.Tensor,
    padded: cute.Tensor,
    cache: cute.Tensor,
    frames: cutlass.Constexpr[int],
    height: cutlass.Constexpr[int],
    width: cutlass.Constexpr[int],
    previous_frames: cutlass.Constexpr[int],
    cache_frames: cutlass.Constexpr[int],
    input_strides: cutlass.Constexpr[tuple[int, ...]],
) -> None:
    """Convert strided C12 NCTHW input and history to padded C16 NTHWC."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    pixel = cutlass.Int64(bid) * 256 + tid
    if pixel < padded.shape[0] // 16:
        w = pixel % (width + 2) - 1
        h = pixel // (width + 2) % (height + 2) - 1
        time = pixel // ((width + 2) * (height + 2)) % (frames + 2) - 2
        batch = pixel // ((frames + 2) * (width + 2) * (height + 2))
        values = cutlass.Array(cutlass.BFloat16, 16, space=cutlass.AddressSpace.rmem)
        values.store(cutlass.vector.full((16,), 0, cutlass.BFloat16), 0)
        if h >= 0 and h < height and w >= 0 and w < width:
            if time >= 0:
                stride_n, stride_c, stride_t, stride_h, stride_w = input_strides
                for channel in cutlass.range_constexpr(12):
                    values[channel] = x[batch * stride_n + channel * stride_c + time * stride_t + h * stride_h + w * stride_w]
            elif cutlass.const_expr(previous_frames > 0):
                if time >= -previous_frames:
                    offset = (((batch * previous_frames + time + previous_frames) * height + h) * width + w) * 12
                    for part in cutlass.range_constexpr(3):
                        values.store(
                            (previous.iterator.raw_ptr() + offset + part * 4).load(count=4, alignment=8),
                            part * 4,
                        )
            if time >= frames - cache_frames:
                cache_offset = (((batch * cache_frames + time - frames + cache_frames) * height + h) * width + w) * 12
                for part in cutlass.range_constexpr(3):
                    (cache.iterator.raw_ptr() + cache_offset + part * 4).store(
                        values.load(part * 4, 4),
                        alignment=8,
                    )
        for part in cutlass.range_constexpr(2):
            (padded.iterator.raw_ptr() + pixel * 16 + part * 8).store(
                values.load(part * 8, 8),
                alignment=16,
            )


_pack_c12_input_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


class CausalConv3dPackLaunch:
    """Compile-time launch description for causal C12 input packing."""

    def __init__(
        self,
        input_shape: tuple[int, int, int, int, int],
        input_strides: tuple[int, ...],
        previous_frames: int,
    ) -> None:
        """Record input geometry, strides, and available history for packing specialization."""
        self.input_shape = input_shape
        self.input_strides = input_strides
        self.previous_frames = previous_frames

    def __repr__(self) -> str:
        """Identify the packing specialization by geometry, history, and input strides."""
        n, _, t, h, w = self.input_shape
        return f"CausalConv3dPack_{n}x{t}x{h}x{w}x12_prev{self.previous_frames}_strides{self.input_strides}"

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        padded: cute.Tensor,
        cache: cute.Tensor,
        stream: cuda_driver.CUstream,
        previous: cute.Tensor = None,
    ) -> None:
        """Launch input/history packing into padded C16 storage and the raw-input cache."""
        n, _, frames, height, width = self.input_shape
        cache_frames = min(2, frames + self.previous_frames)
        _pack_c12_input_kernel(
            x,
            previous,
            padded,
            cache,
            frames,
            height,
            width,
            self.previous_frames,
            cache_frames,
            self.input_strides,
        ).launch(
            grid=((math.prod((n, frames + 2, height + 2, width + 2)) + 255) // 256, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )


@dataclass(frozen=True)
class CausalConv3dConfig:
    """Padded C16 input geometry for the specialized C12-to-C160 convolution."""

    n: int
    t: int
    h: int
    w: int
    max_active_clusters: int

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Return the temporally and spatially padded C16 activation shape."""
        return self.n, self.t, self.h, self.w, 16

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Return the unpadded C160 convolution output shape."""
        return self.n, self.t - 2, self.h - 2, self.w - 2, 160


@cute.jit
def _make_input_tensor_maps(
    config: CausalConv3dConfig,
    weight: cute.Tensor,
    output: cute.Tensor,
) -> tuple[cuda.TensorMap, cuda.TensorMap]:
    """Create tiled-weight and im2col-output tensor maps for the causal convolution."""
    dims, strides = _contiguous_tma_layout((160, 448), cutlass.BFloat16)
    weight_map = cuda.create_tensor_map_tiled(
        global_address=weight.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=dims,
        global_strides=strides,
        box_dims=(64, 160),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    dims, strides = _contiguous_tma_layout(config.output_shape, cutlass.BFloat16)
    output_map = cuda.create_tensor_map_im2col(
        global_address=output.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=dims,
        global_strides=strides,
        lower_corner=(0, 0, 0),
        upper_corner=(0, 0, 0),
        channels_per_pixel=32,
        pixels_per_column=128,
        swizzle=cuda.TensorMapSwizzle.s64b,
    )
    return weight_map, output_map


class CausalConv3dLaunch:
    """Compile-time launch description for the C12-to-C160 causal Conv3D."""

    def __init__(self, config: CausalConv3dConfig) -> None:
        """Record padded geometry and the persistent scheduler's cluster limit."""
        self.config = config

    def __repr__(self) -> str:
        """Identify the convolution specialization by geometry and cluster limit."""
        cfg = self.config
        return f"CausalConv3d_{cfg.n}x{cfg.t}x{cfg.h}x{cfg.w}x12_160_clusters{cfg.max_active_clusters}"

    @cute.jit
    def __call__(
        self,
        padded: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        stream: cuda_driver.CUstream,
    ) -> None:
        """Launch the C12-to-C160 convolution over packed input using a persistent schedule."""
        cfg = self.config
        weight_map, output_map = _make_input_tensor_maps(cfg, weight, output)
        tiles = (cute.ceil_div(math.prod(cfg.output_shape[:-1]), 128), 1, 1)
        scheduler = PersistentTileSchedulerParams(tiles, (1, 1, 1))
        grid = StaticPersistentTileScheduler.get_grid_shape(scheduler, cfg.max_active_clusters)
        _conv3d_postops_kernel(
            "conv_input_c12",
            scheduler,
            (128, 160, 64),
            (128, 160, 16),
            7,
            5,
            2,
            False,
            cfg.output_shape[1:4],
            cfg.n,
            weight_map,
            weight_map,
            output_map,
            packed_x=padded,
            packed_shape=(cfg.n, cfg.t, cfg.h, cfg.w),
        ).launch(
            grid=grid,
            block=(288, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
            smem_merge_branch_allocs=True,
        )


__all__ = [
    "CausalConv3dConfig",
    "CausalConv3dLaunch",
    "CausalConv3dPackLaunch",
]
