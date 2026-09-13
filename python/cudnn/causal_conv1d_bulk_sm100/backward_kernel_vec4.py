# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense vec4 streaming width-four conv backward prototype.

This FE-native implementation was independently derived from the convolution
and SiLU derivatives; it does not copy or call FLA, causal-conv1d, or another
external kernel. It consumes caller-owned workspace and performs no
execute-time allocation.

Each thread owns four adjacent channels and walks one token tile. Three input
and dz values stay in registers, so each output gradient is written without an
atomic. A unique FP32 ``[token_tile, channel, tap]`` partial is followed by one
deterministic reduction kernel. The reducer assigns 16 split lanes to each
channel, then combines their FP32 sums through shared memory.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute

from cudnn.frost.tile_dsl.pointwise import sigmoid

_WIDTH = 4
_VEC = 4
_THREADS = 128
_CHANNELS_PER_CTA = _THREADS * _VEC
_TOKEN_GROUP = 4
_REDUCTION_SPLITS = 16


@cute.jit
def _stream_step(
    x_buffer,
    dy_buffer,
    dx_buffer,
    group_index: cutlass.Constexpr,
    weight,
    x_history,
    dz_history,
    dw_accum,
    dy_mask: cutlass.Float32,
    accumulate_dweight: cutlass.Constexpr,
    produce_dx: cutlass.Constexpr,
    masked: cutlass.Constexpr,
) -> None:
    """Advance one token for four adjacent channels."""

    for lane in cutlass.range_constexpr(_VEC):
        current = x_buffer[group_index, lane].to(cutlass.Float32)
        dy_value = dy_buffer[group_index, lane].to(cutlass.Float32)
        if cutlass.const_expr(masked):
            dy_value = dy_value * dy_mask

        z = x_history[0, lane] * weight[0, lane]
        z = z + x_history[1, lane] * weight[1, lane]
        z = z + x_history[2, lane] * weight[2, lane]
        z = z + current * weight[3, lane]
        gate = sigmoid(z)
        dz = dy_value * (gate + z * (gate * (cutlass.Float32(1.0) - gate)))

        if cutlass.const_expr(accumulate_dweight):
            dw_accum[0, lane] = dw_accum[0, lane] + dz * x_history[0, lane]
            dw_accum[1, lane] = dw_accum[1, lane] + dz * x_history[1, lane]
            dw_accum[2, lane] = dw_accum[2, lane] + dz * x_history[2, lane]
            dw_accum[3, lane] = dw_accum[3, lane] + dz * current

        if cutlass.const_expr(produce_dx):
            dx_value = dz_history[0, lane] * weight[3, lane]
            dx_value = dx_value + dz_history[1, lane] * weight[2, lane]
            dx_value = dx_value + dz_history[2, lane] * weight[1, lane]
            dx_value = dx_value + dz * weight[0, lane]
            dx_buffer[group_index, lane] = dx_value.to(cutlass.BFloat16)

        x_history[0, lane] = x_history[1, lane]
        x_history[1, lane] = x_history[2, lane]
        x_history[2, lane] = current
        dz_history[0, lane] = dz_history[1, lane]
        dz_history[1, lane] = dz_history[2, lane]
        dz_history[2, lane] = dz


@cute.kernel
def _causal_conv1d_bulk_bwd_vec4_kernel(
    x: cute.Tensor,
    weight: cute.Tensor,
    dy: cute.Tensor,
    dx: cute.Tensor,
    dw_partials: cute.Tensor,
    sequence_length: cutlass.Int32,
    tokens_per_cta: cutlass.Int32,
    n_token_tiles: cutlass.Int32,
    n_channels: cutlass.Constexpr,
) -> None:
    """Stream one token tile by four channels per thread."""

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    channel_tile = cutlass.Int32(cute.arch.block_idx()[0])
    token_tile = cutlass.Int32(cute.arch.block_idx()[1])

    channel_vectors = n_channels // _VEC
    data_layout = cute.make_layout(
        (sequence_length, channel_vectors, _VEC),
        stride=(n_channels, _VEC, 1),
    )
    global_x = cute.make_tensor(x.iterator, data_layout)
    global_dy = cute.make_tensor(dy.iterator, data_layout)
    global_dx = cute.make_tensor(dx.iterator, data_layout)
    global_weight = cute.make_tensor(
        weight.iterator,
        cute.make_layout((channel_vectors, _WIDTH * _VEC), stride=(_WIDTH * _VEC, 1)),
    )
    global_partials = cute.make_tensor(
        dw_partials.iterator,
        cute.make_layout(
            (n_token_tiles, n_channels, _WIDTH),
            stride=(n_channels * _WIDTH, _WIDTH, 1),
        ),
    )

    load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(),
        cutlass.BFloat16,
        num_bits_per_copy=_VEC * 16,
        invariant=True,
    )
    channel_vector = channel_tile * (_CHANNELS_PER_CTA // _VEC) + tidx
    token_start = token_tile * tokens_per_cta
    token_end = token_start + tokens_per_cta
    if token_end > sequence_length:
        token_end = sequence_length

    x_history = cute.make_rmem_tensor((3, _VEC), cutlass.Float32)
    dz_history = cute.make_rmem_tensor((3, _VEC), cutlass.Float32)
    dw_accum = cute.make_rmem_tensor((_WIDTH, _VEC), cutlass.Float32)
    weight_values = cute.make_rmem_tensor((_WIDTH, _VEC), cutlass.Float32)
    group_layout = cute.make_layout((_TOKEN_GROUP, _VEC), stride=(_VEC, 1))
    x_buffer = cute.make_rmem_tensor(group_layout, cutlass.BFloat16)
    dy_buffer = cute.make_rmem_tensor(group_layout, cutlass.BFloat16)
    dx_buffer = cute.make_rmem_tensor(group_layout, cutlass.BFloat16)

    for history in cutlass.range_constexpr(3):
        for lane in cutlass.range_constexpr(_VEC):
            x_history[history, lane] = cutlass.Float32(0.0)
            dz_history[history, lane] = cutlass.Float32(0.0)
    for tap in cutlass.range_constexpr(_WIDTH):
        for lane in cutlass.range_constexpr(_VEC):
            dw_accum[tap, lane] = cutlass.Float32(0.0)

    packed_weight = cute.make_rmem_tensor((_WIDTH * _VEC,), weight.element_type)
    cute.autovec_copy(global_weight[(channel_vector, None)], packed_weight)
    for lane in cutlass.range_constexpr(_VEC):
        for tap in cutlass.range_constexpr(_WIDTH):
            weight_values[tap, lane] = packed_weight[_WIDTH * lane + tap].to(cutlass.Float32)

    if token_start >= cutlass.Int32(3):
        for history in cutlass.range_constexpr(3):
            cute.copy(
                load_atom,
                global_x[(token_start - cutlass.Int32(3) + history, channel_vector, None)],
                x_buffer[(0, None)],
            )
            for lane in cutlass.range_constexpr(_VEC):
                x_history[history, lane] = x_buffer[0, lane].to(cutlass.Float32)

    one = cutlass.Float32(1.0)

    # Prime dz[token_start:token_start+3]. The neighboring token tile owns the
    # three dx values preceding this tile.
    for prime in cutlass.range_constexpr(3):
        cute.copy(load_atom, global_x[(token_start + prime, channel_vector, None)], x_buffer[(0, None)])
        cute.copy(load_atom, global_dy[(token_start + prime, channel_vector, None)], dy_buffer[(0, None)])
        _stream_step(
            x_buffer,
            dy_buffer,
            dx_buffer,
            0,
            weight_values,
            x_history,
            dz_history,
            dw_accum,
            one,
            True,
            False,
            False,
        )

    steady_tokens = token_end - token_start - cutlass.Int32(3)
    full_groups = steady_tokens // cutlass.Int32(_TOKEN_GROUP)
    for group in cutlass.range(full_groups, unroll=1):
        base = token_start + cutlass.Int32(3) + group * cutlass.Int32(_TOKEN_GROUP)
        for index in cutlass.range_constexpr(_TOKEN_GROUP):
            cute.copy(load_atom, global_x[(base + index, channel_vector, None)], x_buffer[(index, None)])
            cute.copy(load_atom, global_dy[(base + index, channel_vector, None)], dy_buffer[(index, None)])
        for index in cutlass.range_constexpr(_TOKEN_GROUP):
            _stream_step(
                x_buffer,
                dy_buffer,
                dx_buffer,
                index,
                weight_values,
                x_history,
                dz_history,
                dw_accum,
                one,
                True,
                True,
                False,
            )
        for index in cutlass.range_constexpr(_TOKEN_GROUP):
            cute.autovec_copy(dx_buffer[(index, None)], global_dx[(base + index - cutlass.Int32(3), channel_vector, None)])

    tail_tokens = steady_tokens - full_groups * cutlass.Int32(_TOKEN_GROUP)
    tail_start = token_start + cutlass.Int32(3) + full_groups * cutlass.Int32(_TOKEN_GROUP)
    for index in cutlass.range(tail_tokens, unroll=1):
        cute.copy(load_atom, global_x[(tail_start + index, channel_vector, None)], x_buffer[(0, None)])
        cute.copy(load_atom, global_dy[(tail_start + index, channel_vector, None)], dy_buffer[(0, None)])
        _stream_step(
            x_buffer,
            dy_buffer,
            dx_buffer,
            0,
            weight_values,
            x_history,
            dz_history,
            dw_accum,
            one,
            True,
            True,
            False,
        )
        cute.autovec_copy(dx_buffer[(0, None)], global_dx[(tail_start + index - cutlass.Int32(3), channel_vector, None)])

    # Drain future dz values so the last three dx values owned by this tile
    # receive all four convolution taps. Out-of-range future dy is masked.
    for index in cutlass.range_constexpr(3):
        source_token = token_end + index
        mask = cutlass.Float32(0.0)
        clamped_token = sequence_length - cutlass.Int32(1)
        if source_token < sequence_length:
            mask = cutlass.Float32(1.0)
            clamped_token = source_token
        cute.copy(load_atom, global_x[(clamped_token, channel_vector, None)], x_buffer[(0, None)])
        cute.copy(load_atom, global_dy[(clamped_token, channel_vector, None)], dy_buffer[(0, None)])
        _stream_step(
            x_buffer,
            dy_buffer,
            dx_buffer,
            0,
            weight_values,
            x_history,
            dz_history,
            dw_accum,
            mask,
            False,
            True,
            True,
        )
        cute.autovec_copy(dx_buffer[(0, None)], global_dx[(token_end + index - cutlass.Int32(3), channel_vector, None)])

    channel_partial = cute.make_rmem_tensor((_WIDTH,), cutlass.Float32)
    for lane in cutlass.range_constexpr(_VEC):
        for tap in cutlass.range_constexpr(_WIDTH):
            channel_partial[tap] = dw_accum[tap, lane]
        cute.autovec_copy(
            channel_partial,
            global_partials[(token_tile, channel_vector * cutlass.Int32(_VEC) + lane, None)],
        )


@cute.kernel
def _reduce_dweight_vec4_partials_kernel(
    dw_partials: cute.Tensor,
    dw_accum: cute.Tensor,
    n_token_tiles: cutlass.Int32,
    n_channels: cutlass.Constexpr,
    channels_per_block: cutlass.Constexpr,
) -> None:
    """Reduce one FP32 four-tap vector per output channel."""

    channel_lane = cutlass.Int32(cute.arch.thread_idx()[0])
    split_lane = cutlass.Int32(cute.arch.thread_idx()[1])
    channel = cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(channels_per_block) + channel_lane
    partials = cute.make_tensor(
        dw_partials.iterator,
        cute.make_layout(
            (n_token_tiles, n_channels, _WIDTH),
            stride=(n_channels * _WIDTH, _WIDTH, 1),
        ),
    )
    output = cute.make_tensor(
        dw_accum.iterator,
        cute.make_layout((n_channels, _WIDTH), stride=(_WIDTH, 1)),
    )
    accum = cute.make_rmem_tensor((_WIDTH,), cutlass.Float32)
    values = cute.make_rmem_tensor((_WIDTH,), cutlass.Float32)
    for tap in cutlass.range_constexpr(_WIDTH):
        accum[tap] = cutlass.Float32(0.0)

    # Each split lane owns a strided range. This covers every token tile once
    # for arbitrary n_token_tiles, including n_token_tiles < 16 and a final
    # range shorter than the others.
    for token_tile in cutlass.range(split_lane, n_token_tiles, _REDUCTION_SPLITS, unroll=1):
        cute.autovec_copy(partials[(token_tile, channel, None)], values)
        for tap in cutlass.range_constexpr(_WIDTH):
            accum[tap] = accum[tap] + values[tap]

    smem = cutlass.utils.SmemAllocator()
    split_partials = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout(
            (_REDUCTION_SPLITS, channels_per_block, _WIDTH),
            stride=(channels_per_block * _WIDTH, _WIDTH, 1),
        ),
        byte_alignment=16,
    )
    cute.autovec_copy(accum, split_partials[(split_lane, channel_lane, None)])
    cute.arch.barrier()

    if split_lane == 0:
        for tap in cutlass.range_constexpr(_WIDTH):
            accum[tap] = cutlass.Float32(0.0)
        for reduction_split in cutlass.range_constexpr(_REDUCTION_SPLITS):
            cute.autovec_copy(split_partials[(reduction_split, channel_lane, None)], values)
            for tap in cutlass.range_constexpr(_WIDTH):
                accum[tap] = accum[tap] + values[tap]
        cute.autovec_copy(accum, output[(channel, None)])


class CausalConv1dBulkBackwardVec4Kernel:
    """Launch the dense vec4 streaming kernel and deterministic reduction."""

    def __init__(
        self,
        *,
        sequence_length: int,
        n_channels: int,
        tokens_per_cta: int,
        n_token_tiles: int,
        reduction_threads: int,
    ) -> None:
        if reduction_threads <= 0 or reduction_threads % _REDUCTION_SPLITS != 0:
            raise ValueError(f"reduction_threads must be positive and divisible by {_REDUCTION_SPLITS}, got {reduction_threads}")
        channels_per_reduction_block = reduction_threads // _REDUCTION_SPLITS
        if n_channels % channels_per_reduction_block != 0:
            raise ValueError(f"n_channels must be divisible by {channels_per_reduction_block}, got {n_channels}")
        self.sequence_length = sequence_length
        self.n_channels = n_channels
        self.tokens_per_cta = tokens_per_cta
        self.n_token_tiles = n_token_tiles
        self.reduction_threads = reduction_threads

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        bias: cute.Tensor | None,
        dy: cute.Tensor,
        dx: cute.Tensor,
        dw_accum: cute.Tensor,
        db_accum: cute.Tensor | None,
        dw_partials: cute.Tensor,
        cu_seqlens: cute.Tensor | None,
        initial_state: cute.Tensor | None,
        d_final_state: cute.Tensor | None,
        d_initial_state: cute.Tensor | None,
        packed_tile_map: cute.Tensor | None,
        stream: cuda.CUstream,
    ) -> None:
        _causal_conv1d_bulk_bwd_vec4_kernel(
            x,
            weight,
            dy,
            dx,
            dw_partials,
            cutlass.Int32(self.sequence_length),
            cutlass.Int32(self.tokens_per_cta),
            cutlass.Int32(self.n_token_tiles),
            self.n_channels,
        ).launch(
            grid=(self.n_channels // _CHANNELS_PER_CTA, self.n_token_tiles, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )
        _reduce_dweight_vec4_partials_kernel(
            dw_partials,
            dw_accum,
            cutlass.Int32(self.n_token_tiles),
            self.n_channels,
            self.reduction_threads // _REDUCTION_SPLITS,
        ).launch(
            grid=(self.n_channels // (self.reduction_threads // _REDUCTION_SPLITS), 1, 1),
            block=(self.reduction_threads // _REDUCTION_SPLITS, _REDUCTION_SPLITS, 1),
            stream=stream,
        )
