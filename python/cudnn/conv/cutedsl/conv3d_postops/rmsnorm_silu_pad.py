# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Device algorithm source: pkuwangh/mlsys@72c2384a5c325198d6600906d5835d1ef6faabac
# mlsys/gpu/kernels/wan_vae/mega_wan_vae/kernels/norm.py

"""Standalone RMSNorm, SiLU, padding, and history/cache copies; no convolution."""

import cutlass
from cuda.bindings import driver as cuda_driver
from cutlass import cute

from .kernel import _coordinate_divisor

_THREADS = 128
_VECTOR = 8
_AUX_VECTORS_PER_THREAD = 4


@cute.jit
def _pad_aux_vector(
    previous: cute.Tensor,
    padded: cute.Tensor,
    cache: cute.Tensor,
    vector_idx,
    batches,
    frames,
    height,
    width,
    previous_frames,
    cache_frames,
    channels: cutlass.Constexpr[int],
    has_previous: cutlass.Constexpr[bool],
    divisors,
    wide: cutlass.Constexpr[bool],
) -> None:
    """Write one history or halo vector outside current-frame interiors."""
    _, _, _, padded_plane_div, padded_width_div, auxiliary_div, border_div = divisors
    padded_h = height + 2
    padded_w = width + 2
    history_rows = 2 * padded_h * padded_w
    border_rows = 2 * padded_w + 2 * height
    auxiliary_rows = history_rows + frames * border_rows
    vectors_per_row = channels // _VECTOR
    vectors_per_batch = auxiliary_rows * vectors_per_row
    if vector_idx < batches * vectors_per_batch:
        batch = vector_idx // auxiliary_div
        local_vector = vector_idx % auxiliary_div
        channel = local_vector % vectors_per_row * _VECTOR
        local_row = local_vector // vectors_per_row
        pt = cutlass.Int64(0) if wide else cutlass.Int32(0)
        ph = cutlass.Int64(0) if wide else cutlass.Int32(0)
        pw = cutlass.Int64(0) if wide else cutlass.Int32(0)
        if local_row < history_rows:
            pt = local_row // padded_plane_div
            ph = (local_row % padded_plane_div) // padded_width_div
            pw = (local_row % padded_plane_div) % padded_width_div
        else:
            border = (local_row - history_rows) % border_div
            pt = (local_row - history_rows) // border_div + 2
            if border < 2 * padded_w:
                ph = border // padded_width_div * (height + 1)
                pw = border % padded_width_div
            else:
                ph = (border - 2 * padded_w) // 2 + 1
                pw = (border - 2 * padded_w) % 2 * (width + 1)

        values = cutlass.vector.full((_VECTOR,), 0, padded.element_type)
        valid_spatial = ph > 0 and ph <= height and pw > 0 and pw <= width
        if cutlass.const_expr(has_previous):  # noqa: SIM102 - preserve DSL specialization guard
            if pt < 2 and pt >= 2 - previous_frames and valid_spatial:
                source = (((cutlass.Int64(batch) * previous_frames + pt - 2 + previous_frames) * height + ph - 1) * width + pw - 1) * channels + channel
                values = (previous.iterator.raw_ptr() + source).load(count=_VECTOR, alignment=16)
                if pt - 2 >= frames - cache_frames:
                    cache_offset = (
                        ((cutlass.Int64(batch) * cache_frames + pt - 2 - frames + cache_frames) * height + ph - 1) * width + pw - 1
                    ) * channels + channel
                    (cache.iterator.raw_ptr() + cache_offset).store(values, alignment=16)

        destination = (((cutlass.Int64(batch) * (frames + 2) + pt) * padded_h + ph) * padded_w + pw) * channels + channel
        (padded.iterator.raw_ptr() + destination).store(values, alignment=16)


@cute.jit
def _rmsnorm_silu_current_row(
    x: cute.Tensor,
    gamma: cute.Tensor,
    input_bias: cute.Tensor,
    residual: cute.Tensor,
    residual_bias: cute.Tensor,
    residual_out: cute.Tensor,
    padded: cute.Tensor,
    cache: cute.Tensor,
    input_row: cutlass.Int64,
    geometry,
    cache_frames,
    divisors,
    CHANNELS: cutlass.Constexpr[int],
    HAS_INPUT_BIAS: cutlass.Constexpr[bool],
    HAS_RESIDUAL: cutlass.Constexpr[bool],
    HAS_RESIDUAL_BIAS: cutlass.Constexpr[bool],
) -> None:
    """Vectorize channels while preserving Torch's 32-lane reduction tree.

    C160/C256/C320 use eight physical lanes to represent 32 virtual lanes;
    wider channels use a full warp to bound per-thread retained values. Each lane owns
    four adjacent channels per group. Channels separated by 128 accumulate
    before each virtual lane's four partials are summed. For eight lanes,
    ((group0 + group2) + (group1 + group3)) implements warp offsets 16 and 8;
    physical shuffles finish the tree. This is the same FP32 tree as
    rmsnorm_silu_row(CONTIGUOUS_REDUCTION=True), with the same BF16 cast points.
    Values stay packed in registers between load/reduction and activation/store.
    Canonical row strides and column offsets preserve the checked 8B alignment.
    """
    tid, _, _ = cute.arch.thread_idx()
    LANES = 32 if CHANNELS >= 512 else 8
    VIRTUAL_GROUPS = 32 // LANES
    lane = tid % LANES
    lane_base = tid % 32 // LANES * LANES
    mask = cutlass.Uint32((1 << LANES) - 1) << lane_base
    retained = cutlass.Array(x.element_type, CHANNELS // LANES, space=cutlass.AddressSpace.rmem)
    partials = cutlass.Array(cutlass.Float32, VIRTUAL_GROUPS * 4, space=cutlass.AddressSpace.rmem)
    for group in cutlass.range_constexpr(CHANNELS // (LANES * 4)):
        col = lane * 4 + group * LANES * 4
        offset = cutlass.Int64(input_row) * CHANNELS + col
        value = (x.iterator.raw_ptr() + offset).load(count=4, alignment=8)
        if cutlass.const_expr(HAS_INPUT_BIAS):
            b = (input_bias.iterator.raw_ptr() + col).load(count=4, alignment=8)
            value = (value.to(cutlass.Float32) + b.to(cutlass.Float32)).to(x.element_type)
        if cutlass.const_expr(HAS_RESIDUAL):
            skip = (residual.iterator.raw_ptr() + offset).load(count=4, alignment=8)
            if cutlass.const_expr(HAS_RESIDUAL_BIAS):
                b = (residual_bias.iterator.raw_ptr() + col).load(count=4, alignment=8)
                skip = (skip.to(cutlass.Float32) + b.to(cutlass.Float32)).to(x.element_type)
            value = (value.to(cutlass.Float32) + skip.to(cutlass.Float32)).to(x.element_type)
        if cutlass.const_expr(residual_out is not None):
            (residual_out.iterator.raw_ptr() + offset).store(value, alignment=8)
        retained.store(value, group * 4)
        fp = value.to(cutlass.Float32)
        squares = fp * fp
        if cutlass.const_expr(CHANNELS == 96):
            # Torch uses scalar lanes below C128: c, c+32, c+64.
            # Each physical lane holds four adjacent virtual lanes.
            if cutlass.const_expr(group > 0):
                squares = partials.load(0, 4) + squares
            partials.store(squares, 0)
        else:
            if cutlass.const_expr(group >= VIRTUAL_GROUPS):
                squares = partials.load((group % VIRTUAL_GROUPS) * 4, 4) + squares
            partials.store(squares, (group % VIRTUAL_GROUPS) * 4)
    if cutlass.const_expr(CHANNELS == 96):
        for part in cutlass.range_constexpr(4):
            total = partials[part]
            for offset in (4, 2, 1):
                total = total + cute.arch.shuffle_sync_down(total, offset, mask=mask)
            partials[part] = total
        total = (partials[0] + partials[2]) + (partials[1] + partials[3])
    else:
        groups = cutlass.Array(cutlass.Float32, VIRTUAL_GROUPS, space=cutlass.AddressSpace.rmem)
        for group in cutlass.range_constexpr(VIRTUAL_GROUPS):
            total = partials[group * 4]
            for part in cutlass.range_constexpr(1, 4):
                total = total + partials[group * 4 + part]
            groups[group] = total
        if cutlass.const_expr(LANES == 8):
            total = (groups[0] + groups[2]) + (groups[1] + groups[3])
        else:
            total = groups[0]
        for offset in (16, 8, 4, 2, 1) if LANES == 32 else (4, 2, 1):
            total = total + cute.arch.shuffle_sync_down(total, offset, mask=mask)
    denominator = cute.math.sqrt(cute.arch.shuffle_sync(total, lane_base, mask=mask), fastmath=False)
    if denominator < 1e-12:
        denominator = cutlass.Float32(1e-12)
    frames, height, width = geometry
    volume_div, plane_div, width_div, _, _, _, _ = divisors
    batch = input_row // volume_div
    local_row = input_row % volume_div
    time = local_row // plane_div
    spatial_row = local_row % plane_div
    h = spatial_row // width_div
    w = spatial_row % width_div
    padded_row = ((batch * (frames + 2) + time + 2) * (height + 2) + h + 1) * (width + 2) + w + 1
    cache_row = ((batch * cache_frames + time - frames + cache_frames) * height + h) * width + w
    write_cache = time >= frames - cache_frames
    for group in cutlass.range_constexpr(CHANNELS // (LANES * 4)):
        col = lane * 4 + group * LANES * 4
        value = retained.load(group * 4, 4).to(cutlass.Float32)
        normalized = (value / denominator).to(x.element_type)
        scaled = (normalized.to(cutlass.Float32) * (CHANNELS**0.5)).to(x.element_type)
        g = (gamma.iterator.raw_ptr() + col).load(count=4, alignment=8)
        affine = (scaled.to(cutlass.Float32) * g.to(cutlass.Float32)).to(x.element_type)
        value = affine.to(cutlass.Float32)
        result = (value / (1.0 + cute.math.exp(-value, fastmath=False))).to(x.element_type)
        (padded.iterator.raw_ptr() + cutlass.Int64(padded_row) * CHANNELS + col).store(result, alignment=8)
        if write_cache:
            (cache.iterator.raw_ptr() + cutlass.Int64(cache_row) * CHANNELS + col).store(result, alignment=8)


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
    batches,
    frames,
    height,
    width,
    previous_frames,
    cache_frames,
    channels: cutlass.Constexpr[int],
    has_input_bias: cutlass.Constexpr[bool],
    has_residual: cutlass.Constexpr[bool],
    has_residual_bias: cutlass.Constexpr[bool],
    has_previous: cutlass.Constexpr[bool],
    divisors,
    wide: cutlass.Constexpr[bool],
) -> None:
    """Partition one launch between normalization rows and history/halo vectors."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    row_lanes = 32 if channels >= 512 else 8
    rows = batches * frames * height * width
    current_blocks = (rows + _THREADS // row_lanes - 1) // (_THREADS // row_lanes)
    if bid < current_blocks:
        input_row = (cutlass.Int64(bid) if wide else bid) * (_THREADS // row_lanes) + tid // row_lanes
        if input_row < rows:
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
                (frames, height, width),
                cache_frames,
                divisors,
                channels,
                has_input_bias,
                has_residual,
                has_residual_bias,
            )
    else:
        vector_idx = ((cutlass.Int64(bid) if wide else bid) - current_blocks) * _THREADS * _AUX_VECTORS_PER_THREAD + tid
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
                divisors,
                wide,
            )


_rmsnorm_silu_pad_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


class RmsNormSiluPadLaunch:
    """Compile-time launch description for standalone RMSNorm, SiLU, and padding."""

    def __init__(
        self,
        channels: int,
        wide: bool,
        has_input_bias: bool,
        has_residual: bool,
        has_residual_bias: bool,
    ) -> None:
        """Record activation geometry, history length, and optional bias/residual operations."""
        self.channels = channels
        self.wide = wide
        self.has_input_bias = has_input_bias
        self.has_residual = has_residual
        self.has_residual_bias = has_residual_bias

    def __repr__(self) -> str:
        """Identify the normalization specialization by shape, history, and fused additions."""
        return f"RmsNormSiluPad_c{self.channels}_wide{self.wide}_bias{self.has_input_bias}_res{self.has_residual}_rb{self.has_residual_bias}"

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
        *,
        geometry,
        history: cutlass.Int32,
    ) -> None:
        """Launch normalization, SiLU, padding, and history/cache writes."""
        n, frames, height, width = geometry
        channels = self.channels
        border_rows = 2 * (width + 2) + 2 * height
        auxiliary_vectors = (2 * (height + 2) * (width + 2) + frames * border_rows) * (channels // _VECTOR)
        divisors = tuple(
            _coordinate_divisor(v, self.wide)
            for v in (frames * height * width, height * width, width, (height + 2) * (width + 2), width + 2, auxiliary_vectors, border_rows)
        )
        row_lanes = 32 if channels >= 512 else 8
        rows = n * frames * height * width
        current_blocks = (rows + _THREADS // row_lanes - 1) // (_THREADS // row_lanes)
        padded_h = height + 2
        padded_w = width + 2
        auxiliary_rows = 2 * padded_h * padded_w + frames * (2 * padded_w + 2 * height)
        vectors = n * auxiliary_rows * (channels // _VECTOR)
        vectors_per_cta = _THREADS * _AUX_VECTORS_PER_THREAD
        auxiliary_blocks = (vectors + vectors_per_cta - 1) // vectors_per_cta
        cache_frames = min(2, frames + history)
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
            history,
            cache_frames,
            channels,
            self.has_input_bias,
            self.has_residual,
            self.has_residual_bias,
            previous is not None,
            divisors,
            self.wide,
        ).launch(
            grid=(current_blocks + auxiliary_blocks, 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )


__all__ = ["RmsNormSiluPadLaunch"]
