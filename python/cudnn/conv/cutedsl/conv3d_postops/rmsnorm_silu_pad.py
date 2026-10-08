# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone RMSNorm, SiLU, padding, and history/cache copies; no convolution."""

import cutlass
from cuda.bindings import driver as cuda_driver
from cutlass import cute

_THREADS = 128
_VECTOR = 8
_AUX_VECTORS_PER_THREAD = 4


@cute.jit
def _pad_aux_vector(
    previous: cute.Tensor,
    padded: cute.Tensor,
    cache: cute.Tensor,
    vector_idx: cutlass.Int64,
    batches: cutlass.Constexpr[int],
    frames: cutlass.Constexpr[int],
    height: cutlass.Constexpr[int],
    width: cutlass.Constexpr[int],
    previous_frames: cutlass.Constexpr[int],
    cache_frames: cutlass.Constexpr[int],
    channels: cutlass.Constexpr[int],
    has_previous: cutlass.Constexpr[bool],
) -> None:
    """Write one history or halo vector outside current-frame interiors."""
    padded_h = height + 2
    padded_w = width + 2
    history_rows = 2 * padded_h * padded_w
    border_rows = 2 * padded_w + 2 * height
    auxiliary_rows = history_rows + frames * border_rows
    vectors_per_row = channels // _VECTOR
    vectors_per_batch = auxiliary_rows * vectors_per_row
    if vector_idx < batches * vectors_per_batch:
        batch = vector_idx // vectors_per_batch
        local_vector = cutlass.Int64(vector_idx % vectors_per_batch)
        channel = local_vector % vectors_per_row * _VECTOR
        local_row = local_vector // vectors_per_row
        pt = cutlass.Int64(0)
        ph = cutlass.Int64(0)
        pw = cutlass.Int64(0)
        if local_row < history_rows:
            pt = local_row // (padded_h * padded_w)
            ph = local_row // padded_w % padded_h
            pw = local_row % padded_w
        else:
            border = (local_row - history_rows) % border_rows
            pt = (local_row - history_rows) // border_rows + 2
            if border < 2 * padded_w:
                ph = border // padded_w * (height + 1)
                pw = border % padded_w
            else:
                ph = (border - 2 * padded_w) // 2 + 1
                pw = (border - 2 * padded_w) % 2 * (width + 1)

        values = cutlass.vector.full((_VECTOR,), 0, cutlass.BFloat16)
        valid_spatial = ph > 0 and ph <= height and pw > 0 and pw <= width
        if cutlass.const_expr(has_previous):  # noqa: SIM102 - preserve DSL specialization guard
            if pt < 2 and pt >= 2 - previous_frames and valid_spatial:
                source = (((batch * previous_frames + pt - 2 + previous_frames) * height + ph - 1) * width + pw - 1) * channels + channel
                values = (previous.iterator.raw_ptr() + source).load(count=_VECTOR, alignment=16)
                if pt - 2 >= frames - cache_frames:
                    cache_offset = (((batch * cache_frames + pt - 2 - frames + cache_frames) * height + ph - 1) * width + pw - 1) * channels + channel
                    (cache.iterator.raw_ptr() + cache_offset).store(values, alignment=16)

        destination = (((batch * (frames + 2) + pt) * padded_h + ph) * padded_w + pw) * channels + channel
        (padded.iterator.raw_ptr() + destination).store(values, alignment=16)


@cute.jit
def _rmsnorm_silu_current_row(
    x: cute.Tensor,
    gamma: cute.Tensor,
    input_bias: cute.Tensor,
    residual: cute.Tensor,
    residual_bias: cute.Tensor,
    residual_output: cute.Tensor,
    padded: cute.Tensor,
    cache: cute.Tensor,
    input_row: cutlass.Int64,
    padded_row: cutlass.Int64,
    cache_row: cutlass.Int64,
    write_cache: cutlass.Boolean,
    channels: cutlass.Constexpr[int],
    has_input_bias: cutlass.Constexpr[bool],
    has_residual: cutlass.Constexpr[bool],
    has_residual_bias: cutlass.Constexpr[bool],
) -> None:
    """Normalize one contiguous-channel row and retain values in registers."""
    tid, _, _ = cute.arch.thread_idx()
    lanes = 32 if channels == 640 else 8
    virtual_groups = 32 // lanes
    lane = tid % lanes
    lane_base = tid % 32 // lanes * lanes
    mask = cutlass.Uint32((1 << lanes) - 1) << lane_base
    retained = cutlass.Array(cutlass.BFloat16, channels // lanes, space=cutlass.AddressSpace.rmem)
    partials = cutlass.Array(cutlass.Float32, virtual_groups * 4, space=cutlass.AddressSpace.rmem)

    for group in cutlass.range_constexpr(channels // (lanes * 4)):
        col = lane * 4 + group * lanes * 4
        offset = input_row * channels + col
        value = (x.iterator.raw_ptr() + offset).load(count=4, alignment=8)
        if cutlass.const_expr(has_input_bias):
            bias = (input_bias.iterator.raw_ptr() + col).load(count=4, alignment=8)
            value = (value.to(cutlass.Float32) + bias.to(cutlass.Float32)).to(cutlass.BFloat16)
        if cutlass.const_expr(has_residual):
            skip = (residual.iterator.raw_ptr() + offset).load(count=4, alignment=8)
            if cutlass.const_expr(has_residual_bias):
                bias = (residual_bias.iterator.raw_ptr() + col).load(count=4, alignment=8)
                skip = (skip.to(cutlass.Float32) + bias.to(cutlass.Float32)).to(cutlass.BFloat16)
            value = (value.to(cutlass.Float32) + skip.to(cutlass.Float32)).to(cutlass.BFloat16)
        if cutlass.const_expr(residual_output is not None):
            (residual_output.iterator.raw_ptr() + offset).store(value, alignment=8)

        retained.store(value, group * 4)
        fp32 = value.to(cutlass.Float32)
        squares = fp32 * fp32
        if cutlass.const_expr(group >= virtual_groups):
            squares = partials.load((group % virtual_groups) * 4, 4) + squares
        partials.store(squares, (group % virtual_groups) * 4)

    groups = cutlass.Array(cutlass.Float32, virtual_groups, space=cutlass.AddressSpace.rmem)
    for group in cutlass.range_constexpr(virtual_groups):
        total = partials[group * 4]
        for part in cutlass.range_constexpr(1, 4):
            total = total + partials[group * 4 + part]
        groups[group] = total
    if cutlass.const_expr(lanes == 8):
        total = (groups[0] + groups[2]) + (groups[1] + groups[3])
    else:
        total = groups[0]
    for offset in (16, 8, 4, 2, 1) if lanes == 32 else (4, 2, 1):
        total = total + cute.arch.shuffle_sync_down(total, offset, mask=mask)
    denominator = cute.math.sqrt(cute.arch.shuffle_sync(total, lane_base, mask=mask), fastmath=False)
    if denominator < 1e-12:
        denominator = cutlass.Float32(1e-12)

    for group in cutlass.range_constexpr(channels // (lanes * 4)):
        col = lane * 4 + group * lanes * 4
        value = retained.load(group * 4, 4).to(cutlass.Float32)
        normalized = (value / denominator).to(cutlass.BFloat16)
        scaled = (normalized.to(cutlass.Float32) * (channels**0.5)).to(cutlass.BFloat16)
        affine_scale = (gamma.iterator.raw_ptr() + col).load(count=4, alignment=8)
        affine = (scaled.to(cutlass.Float32) * affine_scale.to(cutlass.Float32)).to(cutlass.BFloat16)
        fp32 = affine.to(cutlass.Float32)
        activated = (fp32 / (1.0 + cute.math.exp(-fp32, fastmath=False))).to(cutlass.BFloat16)
        (padded.iterator.raw_ptr() + padded_row * channels + col).store(activated, alignment=8)
        if write_cache:
            (cache.iterator.raw_ptr() + cache_row * channels + col).store(activated, alignment=8)


@cute.kernel
def _rmsnorm_silu_pad_kernel(
    x: cute.Tensor,
    gamma: cute.Tensor,
    input_bias: cute.Tensor,
    residual: cute.Tensor,
    residual_bias: cute.Tensor,
    residual_output: cute.Tensor,
    previous: cute.Tensor,
    padded: cute.Tensor,
    cache: cute.Tensor,
    batches: cutlass.Constexpr[int],
    frames: cutlass.Constexpr[int],
    height: cutlass.Constexpr[int],
    width: cutlass.Constexpr[int],
    previous_frames: cutlass.Constexpr[int],
    cache_frames: cutlass.Constexpr[int],
    channels: cutlass.Constexpr[int],
    has_input_bias: cutlass.Constexpr[bool],
    has_residual: cutlass.Constexpr[bool],
    has_residual_bias: cutlass.Constexpr[bool],
    has_previous: cutlass.Constexpr[bool],
) -> None:
    """Partition one launch between normalization rows and history/halo vectors."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    row_lanes = 32 if channels == 640 else 8
    rows = batches * frames * height * width
    current_blocks = (rows + _THREADS // row_lanes - 1) // (_THREADS // row_lanes)
    if bid < current_blocks:
        input_row = cutlass.Int64(bid) * (_THREADS // row_lanes) + tid // row_lanes
        if input_row < rows:
            rows_per_batch = frames * height * width
            batch = input_row // rows_per_batch
            local_row = cutlass.Int64(input_row % rows_per_batch)
            time = local_row // (height * width)
            h = local_row // width % height
            w = local_row % width
            padded_row = ((batch * (frames + 2) + time + 2) * (height + 2) + h + 1) * (width + 2) + w + 1
            cache_row = ((batch * cache_frames + time - frames + cache_frames) * height + h) * width + w
            _rmsnorm_silu_current_row(
                x,
                gamma,
                input_bias,
                residual,
                residual_bias,
                residual_output,
                padded,
                cache,
                input_row,
                padded_row,
                cache_row,
                time >= frames - cache_frames,
                channels,
                has_input_bias,
                has_residual,
                has_residual_bias,
            )
    else:
        vector_idx = (cutlass.Int64(bid) - current_blocks) * _THREADS * _AUX_VECTORS_PER_THREAD + tid
        for step in cutlass.range(_AUX_VECTORS_PER_THREAD, unroll=1):
            _pad_aux_vector(
                previous,
                padded,
                cache,
                vector_idx + step * _THREADS,
                batches,
                frames,
                height,
                width,
                previous_frames,
                cache_frames,
                channels,
                has_previous,
            )


_rmsnorm_silu_pad_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


class RmsNormSiluPadLaunch:
    """Compile-time launch description for standalone RMSNorm, SiLU, and padding."""

    def __init__(
        self,
        shape: tuple[int, int, int, int, int],
        previous_frames: int,
        has_input_bias: bool,
        has_residual: bool,
        has_residual_bias: bool,
    ) -> None:
        """Record activation geometry, history length, and optional bias/residual operations."""
        self.shape = shape
        self.previous_frames = previous_frames
        self.has_input_bias = has_input_bias
        self.has_residual = has_residual
        self.has_residual_bias = has_residual_bias

    def __repr__(self) -> str:
        """Identify the normalization specialization by shape, history, and fused additions."""
        n, t, h, w, c = self.shape
        return f"RmsNormSiluPad_{n}x{t}x{h}x{w}x{c}_prev{self.previous_frames}_" f"bias{self.has_input_bias}_res{self.has_residual}_rb{self.has_residual_bias}"

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        gamma: cute.Tensor,
        padded: cute.Tensor,
        cache: cute.Tensor,
        stream: cuda_driver.CUstream,
        input_bias: cute.Tensor = None,
        residual: cute.Tensor = None,
        residual_bias: cute.Tensor = None,
        residual_output: cute.Tensor = None,
        previous: cute.Tensor = None,
    ) -> None:
        """Launch normalization, SiLU, padding, and history/cache writes."""
        n, frames, height, width, channels = self.shape
        row_lanes = 32 if channels == 640 else 8
        rows = n * frames * height * width
        current_blocks = (rows + _THREADS // row_lanes - 1) // (_THREADS // row_lanes)
        padded_h = height + 2
        padded_w = width + 2
        auxiliary_rows = 2 * padded_h * padded_w + frames * (2 * padded_w + 2 * height)
        vectors = n * auxiliary_rows * (channels // _VECTOR)
        vectors_per_cta = _THREADS * _AUX_VECTORS_PER_THREAD
        auxiliary_blocks = (vectors + vectors_per_cta - 1) // vectors_per_cta
        cache_frames = min(2, frames + self.previous_frames)
        _rmsnorm_silu_pad_kernel(
            x,
            gamma,
            input_bias,
            residual,
            residual_bias,
            residual_output,
            previous,
            padded,
            cache,
            n,
            frames,
            height,
            width,
            self.previous_frames,
            cache_frames,
            channels,
            self.has_input_bias,
            self.has_residual,
            self.has_residual_bias,
            self.previous_frames > 0,
        ).launch(
            grid=(current_blocks + auxiliary_blocks, 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )


__all__ = ["RmsNormSiluPadLaunch"]
