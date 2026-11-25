# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Device algorithm source: pkuwangh/mlsys@72c2384a5c325198d6600906d5835d1ef6faabac
# mlsys/gpu/kernels/wan_vae/mega_wan_vae/kernels/conv.py

"""C12 input/history packing and the specialized causal Conv3D."""

import math
from dataclasses import dataclass

import cutlass
from cuda.bindings import driver as cuda_driver
from cutlass import cute
from cutlass.experimental import cuda

from ._config import LaunchPolicy
from ._scheduler import TileSchedule
from ._tf32 import round_tf32
from .kernel import _contiguous_tma_layout, _conv3d_postops_kernel


@cute.kernel
def _pack_c12_input_kernel(
    x: cute.Tensor,
    previous: cute.Tensor,
    padded: cute.Tensor,
    cache: cute.Tensor,
    frames,
    height,
    width,
    previous_frames,
    cache_frames,
    input_strides,
    divisors,
) -> None:
    """Convert strided C12 NCTHW input and history to padded C16 NTHWC."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    pixel = bid * 256 + tid
    volume_div, plane_div, width_div = divisors
    if pixel < padded.shape[0] // 16:
        w = pixel % width_div - 1
        h = (pixel % plane_div) // width_div - 1
        time = (pixel % volume_div) // plane_div - 2
        batch = pixel // volume_div
        values = cutlass.Array(x.element_type, 16, space=cutlass.AddressSpace.rmem)
        values.store(cutlass.vector.full((16,), 0, x.element_type), 0)
        if h >= 0 and h < height and w >= 0 and w < width:
            if time >= 0:
                stride_n, stride_c, stride_t, stride_h, stride_w = input_strides
                for channel in cutlass.range_constexpr(12):
                    values[channel] = x[batch * stride_n + channel * stride_c + time * stride_t + h * stride_h + w * stride_w]
            elif cutlass.const_expr(previous is not None):
                if time >= -previous_frames:
                    offset = (((cutlass.Int64(batch) * previous_frames + time + previous_frames) * height + h) * width + w) * 12
                    for part in cutlass.range_constexpr(3):
                        values.store(
                            (previous.iterator.raw_ptr() + offset + part * 4).load(count=4, alignment=8),
                            part * 4,
                        )
            if time >= frames - cache_frames:
                cache_offset = (((cutlass.Int64(batch) * cache_frames + time - frames + cache_frames) * height + h) * width + w) * 12
                for part in cutlass.range_constexpr(3):
                    (cache.iterator.raw_ptr() + cache_offset + part * 4).store(
                        values.load(part * 4, 4),
                        alignment=8,
                    )
        if cutlass.const_expr(x.element_type == cutlass.Float32):
            values.store(round_tf32(values.load(0, 16)), 0)
        for part in cutlass.range_constexpr(2):
            (padded.iterator.raw_ptr() + cutlass.Int64(pixel) * 16 + part * 8).store(
                values.load(part * 8, 8),
                alignment=16,
            )


_pack_c12_input_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


class CausalConv3dPackLaunch:
    """Compile-time launch description for causal C12 input packing."""

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        padded: cute.Tensor,
        cache: cute.Tensor,
        stream: cuda_driver.CUstream,
        previous: cute.Tensor = None,
        *,
        geometry,
        input_strides,
        history: cutlass.Int32,
    ) -> None:
        """Launch input/history packing into padded C16 storage and the raw-input cache."""
        n, frames, height, width = geometry
        divisors = tuple(cute.fast_divmod_create_divisor(v) for v in ((frames + 2) * (height + 2) * (width + 2), (height + 2) * (width + 2), width + 2))
        cache_frames = min(2, frames + history)
        _pack_c12_input_kernel(
            x,
            previous,
            padded,
            cache,
            frames,
            height,
            width,
            history,
            cache_frames,
            input_strides,
            divisors,
        ).launch(
            grid=((math.prod((n, frames + 2, height + 2, width + 2)) + 255) // 256, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )


@dataclass(frozen=True)
class CausalConv3dConfig:
    """Prepared input dimensions; the convolution itself applies no padding."""

    n: int = 1
    t: int = 3
    h: int = 7
    w: int = 9
    ctas: int = 148

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.n, self.t, self.h, self.w, 16

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.n, self.t - 2, self.h - 2, self.w - 2, 160


def _make_input_tensor_maps(
    cfg: CausalConv3dConfig,
    weight: cute.Tensor,
    output: cute.Tensor,
) -> tuple[cuda.TensorMap, cuda.TensorMap]:
    """Build input-convolution tensor maps inside the JIT host wrapper."""
    dims, strides = _contiguous_tma_layout((160, 448), cutlass.BFloat16)
    tma_b_desc = cuda.create_tensor_map_tiled(
        global_address=weight.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=dims,
        global_strides=strides,
        box_dims=(64, 160),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    dims, strides = _contiguous_tma_layout(cfg.output_shape, cutlass.BFloat16)
    tma_c_desc = cuda.create_tensor_map_im2col(
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
    return tma_b_desc, tma_c_desc


class CausalConv3dLaunch:
    def __init__(self, ctas: int) -> None:
        self.ctas = ctas

    def __repr__(self) -> str:
        return f"CausalConv3d_ctas{self.ctas}"

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        geometry,
        stream: cuda_driver.CUstream,
    ) -> None:
        cfg = CausalConv3dConfig(*geometry, ctas=self.ctas)
        tma_b_desc, tma_c_desc = _make_input_tensor_maps(cfg, weight, output)
        scheduler = TileSchedule(tiles_n=1, cluster_m=1)
        clusters_m = (math.prod(cfg.output_shape[:-1]) + 127) // 128
        policy = LaunchPolicy.for_input()
        z, h, w = cfg.output_shape[1:4]
        _conv3d_postops_kernel(
            "conv_input_c12",
            scheduler,
            (128, 160, 64),
            (128, 160, 16),
            7,
            policy,
            False,
            (z, h, w),
            cfg.n,
            tma_b_desc,
            tma_b_desc,
            tma_c_desc,
            packed_x=x,
            packed_shape=geometry,
            runtime_clusters_m=cute.fast_divmod_create_divisor(clusters_m),
            volume_div=cute.fast_divmod_create_divisor(z * h * w),
            plane_div=cute.fast_divmod_create_divisor(h * w),
            width_div=cute.fast_divmod_create_divisor(w),
        ).launch(
            grid=(1, 1, cutlass.min(clusters_m, self.ctas)),
            block=(policy.threads, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
            smem_merge_branch_allocs=True,
        )
