# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness-first SM100 bulk causal-convolution forward kernel.

This is an independent CuTe DSL implementation of the width-four, BF16,
depthwise causal-convolution contract documented by cuDNN Frontend.  FLA's
short-convolution module and Dao-AILab's public causal-conv1d API were consulted
only as semantic and interface references; no external kernel source is
included here.

The kernel consumes contiguous, flattened ``[total_tokens, D]`` input and
output views.  A dense launch derives sequence boundaries from the host-provided
``tokens_per_sequence`` scalar.  A packed launch reads ``cu_seqlens`` in the
kernel, so neither compilation nor execution needs a device-to-host metadata
read.  Optional initial and final states use the full ``[N, D, 4]`` layout that
can be handed directly to the decode-update primitive.

The schedule deliberately stays small and auditable.  Channel extents divisible
by eight use a 128-thread fast path in which each thread owns eight adjacent
channels and moves each input/output row as one aligned 16-byte transaction.
The fallback keeps the original scalar 256-channel schedule for arbitrary
channel tails.  Both schedules retain a rolling three-value causal window, so
every subsequent output loads only the current token.  The CTA derives its
first sequence interval once and advances it whenever the token loop crosses a
boundary, resetting the window from that sequence's initial state or zeros.
Thus a packed tile may safely straddle any number of sequences.  Packed
launches first run a one-CTA device validator which traps unless the prefix
sums start at zero, end at the runtime token total, and are strictly increasing.

Callers must provide non-overlapping input, output, and state storage.  In
particular, in-place output is unsafe because token CTAs execute independently
and may still need earlier input tokens.
"""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.arch.nvvm_wrappers import inline_ptx

from cudnn.frost.tile_dsl.pointwise import (
    f16x2_to_f32,
    ffma2,
    fmul2,
    fp32_to_fp16,
    opaque_f32_zero,
    sigmoid,
    sigmoid2,
)

THREADS = 256
TOKENS_PER_CTA = 16
VEC_THREADS = 128
VEC_CHANNELS_PER_THREAD = 8
VEC_CHANNELS_PER_CTA = VEC_THREADS * VEC_CHANNELS_PER_THREAD
VEC_TOKENS_PER_CTA = 8
WIDTH = 4


@cute.jit
def _load_bf16x8(address):
    """Load eight aligned BF16 values as four packed words."""

    word_0, word_1, word_2, word_3 = inline_ptx(
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        write_only_types=[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32],
        read_only_args=[address],
    )
    value_0, value_1 = f16x2_to_f32(word_0, dtype=cutlass.BFloat16)
    value_2, value_3 = f16x2_to_f32(word_1, dtype=cutlass.BFloat16)
    value_4, value_5 = f16x2_to_f32(word_2, dtype=cutlass.BFloat16)
    value_6, value_7 = f16x2_to_f32(word_3, dtype=cutlass.BFloat16)
    return value_0, value_1, value_2, value_3, value_4, value_5, value_6, value_7


@cute.jit
def _store_bf16x8(address, value_0, value_1, value_2, value_3, value_4, value_5, value_6, value_7):
    """Store eight BF16 values with one aligned 16-byte transaction."""

    word_0 = fp32_to_fp16(value_0, value_1, dtype=cutlass.BFloat16)
    word_1 = fp32_to_fp16(value_2, value_3, dtype=cutlass.BFloat16)
    word_2 = fp32_to_fp16(value_4, value_5, dtype=cutlass.BFloat16)
    word_3 = fp32_to_fp16(value_6, value_7, dtype=cutlass.BFloat16)
    inline_ptx(
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        read_only_args=[address, word_0, word_1, word_2, word_3],
    )


@cute.kernel
def _validate_cu_seqlens_kernel(
    cu_seqlens: cute.Tensor,
    num_sequences: cutlass.Int32,
    total_tokens: cutlass.Int32,
) -> None:
    """Fail closed on malformed packed boundaries without a host read.

    The CUDA error is asynchronous and, like the decode-update index guards,
    is intentionally implemented with PTX ``trap``.  The validator and the
    convolution launch on the same stream, so an invalid metadata launch cannot
    proceed into tensor reads on that stream.
    """

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    if tidx == cutlass.Int32(0):
        inline_ptx(
            "trap;",
            predicate=cutlass.Int32(cu_seqlens[0]) != cutlass.Int32(0),
        )
        inline_ptx(
            "trap;",
            predicate=cutlass.Int32(cu_seqlens[num_sequences]) != total_tokens,
        )

    boundary = tidx
    while boundary < num_sequences:
        left = cutlass.Int32(cu_seqlens[boundary])
        right = cutlass.Int32(cu_seqlens[boundary + cutlass.Int32(1)])
        inline_ptx("trap;", predicate=right <= left)
        boundary += cutlass.Int32(THREADS)


@cute.kernel
def _causal_conv1d_bulk_fwd_kernel(
    x: cute.Tensor,
    weight: cute.Tensor,
    initial_state: Optional[cute.Tensor],
    cu_seqlens: Optional[cute.Tensor],
    output: cute.Tensor,
    final_state: Optional[cute.Tensor],
    num_sequences: cutlass.Int32,
    tokens_per_sequence: cutlass.Int32,
    n_channels: cutlass.Int32,
) -> None:
    """Emit one token tile by 256 channels and final states at sequence ends.

    ``cu_seqlens`` is ``None``-specialized for dense input.  For packed input,
    lane zero of each warp performs one device-side binary search at the tile
    start and broadcasts the resulting sequence interval.  The token loop
    advances that interval at boundaries.  Every causal read is guarded by the
    local position inside the current interval, preventing reads from a
    preceding packed sequence.  A channel thread keeps its four weights and a
    three-value causal window in registers across the token tile.
    """

    token_tile = cutlass.Int32(cute.arch.block_idx()[0])
    channel_tile = cutlass.Int32(cute.arch.block_idx()[1])
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    channel = channel_tile * cutlass.Int32(THREADS) + tidx
    first_token = token_tile * cutlass.Int32(TOKENS_PER_CTA)
    total_tokens = cutlass.Int32(x.shape[0])

    weight_0 = cutlass.Float32(0.0)
    weight_1 = cutlass.Float32(0.0)
    weight_2 = cutlass.Float32(0.0)
    weight_3 = cutlass.Float32(0.0)
    if channel < n_channels:
        # Vectorize the naturally contiguous four-tap weight row.  Its address
        # is always eight-byte aligned for contiguous BF16 [D, 4] storage.
        weight_element = cutlass.Int64(channel) * cutlass.Int64(weight.stride[0])
        weight_address = weight.iterator.toint() + weight_element * cutlass.Int64(2)
        weight_01, weight_23 = inline_ptx(
            "ld.global.v2.b32 {$0, $1}, [$2];",
            write_only_types=[cutlass.Int32, cutlass.Int32],
            read_only_args=[weight_address],
        )
        weight_0, weight_1 = f16x2_to_f32(weight_01, dtype=cutlass.BFloat16)
        weight_2, weight_3 = f16x2_to_f32(weight_23, dtype=cutlass.BFloat16)

    sequence = cutlass.Int32(0)
    sequence_start = cutlass.Int32(0)
    sequence_end = cutlass.Int32(0)
    if cutlass.const_expr(cu_seqlens is None):
        sequence = first_token // tokens_per_sequence
        sequence_start = sequence * tokens_per_sequence
        sequence_end = sequence_start + tokens_per_sequence
    else:
        # All lanes in a warp process the same token tile.  Search once per
        # warp and broadcast before the channel-tail predicate, so all lanes
        # named by shuffle_sync's mask participate.
        if cute.arch.lane_idx() == cutlass.Int32(0):
            lower = cutlass.Int32(0)
            upper = num_sequences
            while lower + cutlass.Int32(1) < upper:
                middle = (lower + upper) // cutlass.Int32(2)
                if first_token < cutlass.Int32(cu_seqlens[middle]):
                    upper = middle
                else:
                    lower = middle
            sequence = lower
            sequence_start = cutlass.Int32(cu_seqlens[sequence])
            sequence_end = cutlass.Int32(cu_seqlens[sequence + cutlass.Int32(1)])

        sequence = cute.arch.shuffle_sync(sequence, 0)
        sequence_start = cute.arch.shuffle_sync(sequence_start, 0)
        sequence_end = cute.arch.shuffle_sync(sequence_end, 0)

    history_0 = cutlass.Float32(0.0)
    history_1 = cutlass.Float32(0.0)
    history_2 = cutlass.Float32(0.0)
    if channel < n_channels:
        local_first_token = first_token - sequence_start

        # Initialize the three visible values immediately preceding the tile.
        # Near a sequence start, the missing prefix comes from initial-state
        # lanes 1..3; without initial state it remains zero.
        if local_first_token >= cutlass.Int32(3):
            history_0 = x[first_token - cutlass.Int32(3), channel].to(cutlass.Float32)
        elif cutlass.const_expr(initial_state is not None):
            history_0 = initial_state[sequence, channel, local_first_token + cutlass.Int32(1)].to(cutlass.Float32)

        if local_first_token >= cutlass.Int32(2):
            history_1 = x[first_token - cutlass.Int32(2), channel].to(cutlass.Float32)
        elif cutlass.const_expr(initial_state is not None):
            history_1 = initial_state[sequence, channel, local_first_token + cutlass.Int32(2)].to(cutlass.Float32)

        if local_first_token >= cutlass.Int32(1):
            history_2 = x[first_token - cutlass.Int32(1), channel].to(cutlass.Float32)
        elif cutlass.const_expr(initial_state is not None):
            history_2 = initial_state[sequence, channel, cutlass.Int32(3)].to(cutlass.Float32)

    tile_end = first_token + cutlass.Int32(TOKENS_PER_CTA)
    if tile_end > total_tokens:
        tile_end = total_tokens

    token = first_token
    while token < tile_end:
        # Strictly positive sequence lengths guarantee that incrementing one
        # interval is sufficient when the unit-stride token loop hits an end.
        if token == sequence_end:
            sequence += cutlass.Int32(1)
            sequence_start = sequence_end
            if cutlass.const_expr(cu_seqlens is None):
                sequence_end += tokens_per_sequence
            else:
                sequence_end = cutlass.Int32(cu_seqlens[sequence + cutlass.Int32(1)])

            if channel < n_channels:
                history_0 = cutlass.Float32(0.0)
                history_1 = cutlass.Float32(0.0)
                history_2 = cutlass.Float32(0.0)
                if cutlass.const_expr(initial_state is not None):
                    history_0 = initial_state[sequence, channel, cutlass.Int32(1)].to(cutlass.Float32)
                    history_1 = initial_state[sequence, channel, cutlass.Int32(2)].to(cutlass.Float32)
                    history_2 = initial_state[sequence, channel, cutlass.Int32(3)].to(cutlass.Float32)

        if channel < n_channels:
            current = x[token, channel].to(cutlass.Float32)
            acc = history_0 * weight_0
            acc = acc + history_1 * weight_1
            acc = acc + history_2 * weight_2
            acc = acc + current * weight_3
            output[token, channel] = (acc * sigmoid(acc)).to(cutlass.BFloat16)

            if cutlass.const_expr(final_state is not None):
                if token + cutlass.Int32(1) == sequence_end:
                    final_state[sequence, channel, cutlass.Int32(0)] = history_0.to(cutlass.BFloat16)
                    final_state[sequence, channel, cutlass.Int32(1)] = history_1.to(cutlass.BFloat16)
                    final_state[sequence, channel, cutlass.Int32(2)] = history_2.to(cutlass.BFloat16)
                    final_state[sequence, channel, cutlass.Int32(3)] = current.to(cutlass.BFloat16)

            history_0 = history_1
            history_1 = history_2
            history_2 = current

        token += cutlass.Int32(1)


@cute.kernel
def _causal_conv1d_bulk_vec8_fwd_kernel(
    x: cute.Tensor,
    weight: cute.Tensor,
    initial_state: Optional[cute.Tensor],
    cu_seqlens: Optional[cute.Tensor],
    output: cute.Tensor,
    final_state: Optional[cute.Tensor],
    num_sequences: cutlass.Int32,
    tokens_per_sequence: cutlass.Int32,
    n_channels: cutlass.Int32,
) -> None:
    """Emit eight-token tiles with one aligned eight-channel vector per thread.

    This specialization is launched only when the compile-time channel extent
    is divisible by eight.  The API guarantees 16-byte base alignment and all
    row strides are then multiples of eight BF16 elements.  Consequently every
    valid vector below is aligned; ``channel_base < n_channels`` also proves the
    entire eight-channel group is in bounds.
    """

    token_tile = cutlass.Int32(cute.arch.block_idx()[0])
    channel_tile = cutlass.Int32(cute.arch.block_idx()[1])
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    channel_base = channel_tile * cutlass.Int32(VEC_CHANNELS_PER_CTA) + tidx * cutlass.Int32(VEC_CHANNELS_PER_THREAD)
    first_token = token_tile * cutlass.Int32(VEC_TOKENS_PER_CTA)
    total_tokens = cutlass.Int32(x.shape[0])
    valid_channel_group = channel_base < n_channels

    # Four FP32 vectors are the four causal taps for eight channels.  Weight
    # storage is [D, 4], so each 16-byte load naturally covers two complete
    # adjacent channel rows and is transposed into the tap-major registers.
    weight_0 = cutlass.Array(cutlass.Float32, VEC_CHANNELS_PER_THREAD, alignment=16)
    weight_1 = cutlass.Array(cutlass.Float32, VEC_CHANNELS_PER_THREAD, alignment=16)
    weight_2 = cutlass.Array(cutlass.Float32, VEC_CHANNELS_PER_THREAD, alignment=16)
    weight_3 = cutlass.Array(cutlass.Float32, VEC_CHANNELS_PER_THREAD, alignment=16)
    zero = opaque_f32_zero()
    for channel_offset in cutlass.range_constexpr(VEC_CHANNELS_PER_THREAD):
        weight_0[channel_offset] = zero
        weight_1[channel_offset] = zero
        weight_2[channel_offset] = zero
        weight_3[channel_offset] = zero

    if valid_channel_group:
        for channel_pair in cutlass.range_constexpr(VEC_CHANNELS_PER_THREAD // 2):
            pair_channel = channel_base + cutlass.Int32(2 * channel_pair)
            weight_element = cutlass.Int64(pair_channel) * cutlass.Int64(weight.stride[0])
            weight_address = weight.iterator.toint() + weight_element * cutlass.Int64(2)
            values = _load_bf16x8(weight_address)
            channel_offset = 2 * channel_pair
            weight_0[channel_offset] = values[0]
            weight_1[channel_offset] = values[1]
            weight_2[channel_offset] = values[2]
            weight_3[channel_offset] = values[3]
            weight_0[channel_offset + 1] = values[4]
            weight_1[channel_offset + 1] = values[5]
            weight_2[channel_offset + 1] = values[6]
            weight_3[channel_offset + 1] = values[7]

    sequence = cutlass.Int32(0)
    sequence_start = cutlass.Int32(0)
    sequence_end = cutlass.Int32(0)
    if cutlass.const_expr(cu_seqlens is None):
        sequence = first_token // tokens_per_sequence
        sequence_start = sequence * tokens_per_sequence
        sequence_end = sequence_start + tokens_per_sequence
    else:
        # Search and shuffle before the channel predicate: even threads whose
        # vector falls beyond the final channel tile must participate in the
        # full-warp shuffle mask.
        if cute.arch.lane_idx() == cutlass.Int32(0):
            lower = cutlass.Int32(0)
            upper = num_sequences
            while lower + cutlass.Int32(1) < upper:
                middle = (lower + upper) // cutlass.Int32(2)
                if first_token < cutlass.Int32(cu_seqlens[middle]):
                    upper = middle
                else:
                    lower = middle
            sequence = lower
            sequence_start = cutlass.Int32(cu_seqlens[sequence])
            sequence_end = cutlass.Int32(cu_seqlens[sequence + cutlass.Int32(1)])

        sequence = cute.arch.shuffle_sync(sequence, 0)
        sequence_start = cute.arch.shuffle_sync(sequence_start, 0)
        sequence_end = cute.arch.shuffle_sync(sequence_end, 0)

    history_0 = cutlass.Array(cutlass.Float32, VEC_CHANNELS_PER_THREAD, alignment=16)
    history_1 = cutlass.Array(cutlass.Float32, VEC_CHANNELS_PER_THREAD, alignment=16)
    history_2 = cutlass.Array(cutlass.Float32, VEC_CHANNELS_PER_THREAD, alignment=16)
    current = cutlass.Array(cutlass.Float32, VEC_CHANNELS_PER_THREAD, alignment=16)
    output_values = cutlass.Array(cutlass.Float32, VEC_CHANNELS_PER_THREAD, alignment=16)
    for channel_offset in cutlass.range_constexpr(VEC_CHANNELS_PER_THREAD):
        history_0[channel_offset] = zero
        history_1[channel_offset] = zero
        history_2[channel_offset] = zero
        current[channel_offset] = zero
        output_values[channel_offset] = zero

    if valid_channel_group:
        local_first_token = first_token - sequence_start

        if local_first_token >= cutlass.Int32(3):
            x_element = cutlass.Int64(first_token - cutlass.Int32(3)) * cutlass.Int64(x.stride[0]) + cutlass.Int64(channel_base)
            values = _load_bf16x8(x.iterator.toint() + x_element * cutlass.Int64(2))
            for channel_offset in cutlass.range_constexpr(VEC_CHANNELS_PER_THREAD):
                history_0[channel_offset] = values[channel_offset]

        if local_first_token >= cutlass.Int32(2):
            x_element = cutlass.Int64(first_token - cutlass.Int32(2)) * cutlass.Int64(x.stride[0]) + cutlass.Int64(channel_base)
            values = _load_bf16x8(x.iterator.toint() + x_element * cutlass.Int64(2))
            for channel_offset in cutlass.range_constexpr(VEC_CHANNELS_PER_THREAD):
                history_1[channel_offset] = values[channel_offset]

        if local_first_token >= cutlass.Int32(1):
            x_element = cutlass.Int64(first_token - cutlass.Int32(1)) * cutlass.Int64(x.stride[0]) + cutlass.Int64(channel_base)
            values = _load_bf16x8(x.iterator.toint() + x_element * cutlass.Int64(2))
            for channel_offset in cutlass.range_constexpr(VEC_CHANNELS_PER_THREAD):
                history_2[channel_offset] = values[channel_offset]

        # A single 16-byte state load per channel pair provides all four taps.
        # Only the missing prefix is copied; input rows already loaded above
        # remain authoritative farther into a sequence.
        if cutlass.const_expr(initial_state is not None):
            if local_first_token < cutlass.Int32(3):
                for channel_pair in cutlass.range_constexpr(VEC_CHANNELS_PER_THREAD // 2):
                    pair_channel = channel_base + cutlass.Int32(2 * channel_pair)
                    state_element = cutlass.Int64(sequence) * cutlass.Int64(initial_state.stride[0]) + cutlass.Int64(pair_channel) * cutlass.Int64(
                        initial_state.stride[1]
                    )
                    values = _load_bf16x8(initial_state.iterator.toint() + state_element * cutlass.Int64(2))
                    channel_offset = 2 * channel_pair
                    if local_first_token == cutlass.Int32(0):
                        history_0[channel_offset] = values[1]
                        history_1[channel_offset] = values[2]
                        history_2[channel_offset] = values[3]
                        history_0[channel_offset + 1] = values[5]
                        history_1[channel_offset + 1] = values[6]
                        history_2[channel_offset + 1] = values[7]
                    elif local_first_token == cutlass.Int32(1):
                        history_0[channel_offset] = values[2]
                        history_1[channel_offset] = values[3]
                        history_0[channel_offset + 1] = values[6]
                        history_1[channel_offset + 1] = values[7]
                    else:
                        history_0[channel_offset] = values[3]
                        history_0[channel_offset + 1] = values[7]

    tile_end = first_token + cutlass.Int32(VEC_TOKENS_PER_CTA)
    if tile_end > total_tokens:
        tile_end = total_tokens

    # This must remain a genuine runtime loop.  A constexpr-bounded Python
    # while expands every token body and sharply inflates PTX/SASS and register
    # liveness for larger token tiles.
    for token in cutlass.range(first_token, tile_end, 1, unroll=1):
        if token == sequence_end:
            sequence += cutlass.Int32(1)
            sequence_start = sequence_end
            if cutlass.const_expr(cu_seqlens is None):
                sequence_end += tokens_per_sequence
            else:
                sequence_end = cutlass.Int32(cu_seqlens[sequence + cutlass.Int32(1)])

            if valid_channel_group:
                for channel_offset in cutlass.range_constexpr(VEC_CHANNELS_PER_THREAD):
                    history_0[channel_offset] = zero
                    history_1[channel_offset] = zero
                    history_2[channel_offset] = zero
                if cutlass.const_expr(initial_state is not None):
                    for channel_pair in cutlass.range_constexpr(VEC_CHANNELS_PER_THREAD // 2):
                        pair_channel = channel_base + cutlass.Int32(2 * channel_pair)
                        state_element = cutlass.Int64(sequence) * cutlass.Int64(initial_state.stride[0]) + cutlass.Int64(pair_channel) * cutlass.Int64(
                            initial_state.stride[1]
                        )
                        values = _load_bf16x8(initial_state.iterator.toint() + state_element * cutlass.Int64(2))
                        channel_offset = 2 * channel_pair
                        history_0[channel_offset] = values[1]
                        history_1[channel_offset] = values[2]
                        history_2[channel_offset] = values[3]
                        history_0[channel_offset + 1] = values[5]
                        history_1[channel_offset + 1] = values[6]
                        history_2[channel_offset + 1] = values[7]

        if valid_channel_group:
            x_element = cutlass.Int64(token) * cutlass.Int64(x.stride[0]) + cutlass.Int64(channel_base)
            values = _load_bf16x8(x.iterator.toint() + x_element * cutlass.Int64(2))
            for channel_offset in cutlass.range_constexpr(VEC_CHANNELS_PER_THREAD):
                current[channel_offset] = values[channel_offset]

            for channel_pair in cutlass.range_constexpr(VEC_CHANNELS_PER_THREAD // 2):
                lo = 2 * channel_pair
                hi = lo + 1
                acc_lo, acc_hi = fmul2(history_0[lo], history_0[hi], weight_0[lo], weight_0[hi])
                acc_lo, acc_hi = ffma2(history_1[lo], history_1[hi], weight_1[lo], weight_1[hi], acc_lo, acc_hi)
                acc_lo, acc_hi = ffma2(history_2[lo], history_2[hi], weight_2[lo], weight_2[hi], acc_lo, acc_hi)
                acc_lo, acc_hi = ffma2(current[lo], current[hi], weight_3[lo], weight_3[hi], acc_lo, acc_hi)
                gate_lo, gate_hi = sigmoid2(acc_lo, acc_hi)
                output_values[lo], output_values[hi] = fmul2(acc_lo, acc_hi, gate_lo, gate_hi)

            output_element = cutlass.Int64(token) * cutlass.Int64(output.stride[0]) + cutlass.Int64(channel_base)
            _store_bf16x8(
                output.iterator.toint() + output_element * cutlass.Int64(2),
                output_values[0],
                output_values[1],
                output_values[2],
                output_values[3],
                output_values[4],
                output_values[5],
                output_values[6],
                output_values[7],
            )

            if cutlass.const_expr(final_state is not None):
                if token + cutlass.Int32(1) == sequence_end:
                    for channel_pair in cutlass.range_constexpr(VEC_CHANNELS_PER_THREAD // 2):
                        pair_channel = channel_base + cutlass.Int32(2 * channel_pair)
                        state_element = cutlass.Int64(sequence) * cutlass.Int64(final_state.stride[0]) + cutlass.Int64(pair_channel) * cutlass.Int64(
                            final_state.stride[1]
                        )
                        channel_offset = 2 * channel_pair
                        _store_bf16x8(
                            final_state.iterator.toint() + state_element * cutlass.Int64(2),
                            history_0[channel_offset],
                            history_1[channel_offset],
                            history_2[channel_offset],
                            current[channel_offset],
                            history_0[channel_offset + 1],
                            history_1[channel_offset + 1],
                            history_2[channel_offset + 1],
                            current[channel_offset + 1],
                        )

            for channel_offset in cutlass.range_constexpr(VEC_CHANNELS_PER_THREAD):
                history_0[channel_offset] = history_1[channel_offset]
                history_1[channel_offset] = history_2[channel_offset]
                history_2[channel_offset] = current[channel_offset]


class CausalConv1dBulkForwardKernel:
    """Host launcher for the fixed BF16, width-four SM100 forward slice."""

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        initial_state: Optional[cute.Tensor],
        cu_seqlens: Optional[cute.Tensor],
        output: cute.Tensor,
        final_state: Optional[cute.Tensor],
        num_sequences: cutlass.Int32,
        tokens_per_sequence: cutlass.Int32,
        n_channels: cutlass.Int32,
        stream: cuda.CUstream,
    ) -> None:
        if cutlass.const_expr(cu_seqlens is not None):
            _validate_cu_seqlens_kernel(
                cu_seqlens,
                num_sequences,
                cutlass.Int32(x.shape[0]),
            ).launch(
                grid=(1, 1, 1),
                block=(THREADS, 1, 1),
                stream=stream,
            )

        if cutlass.const_expr(x.shape[1] % VEC_CHANNELS_PER_THREAD == 0):
            _causal_conv1d_bulk_vec8_fwd_kernel(
                x,
                weight,
                initial_state,
                cu_seqlens,
                output,
                final_state,
                num_sequences,
                tokens_per_sequence,
                n_channels,
            ).launch(
                grid=(cute.ceil_div(x.shape[0], VEC_TOKENS_PER_CTA), cute.ceil_div(x.shape[1], VEC_CHANNELS_PER_CTA), 1),
                block=(VEC_THREADS, 1, 1),
                stream=stream,
            )
        else:
            _causal_conv1d_bulk_fwd_kernel(
                x,
                weight,
                initial_state,
                cu_seqlens,
                output,
                final_state,
                num_sequences,
                tokens_per_sequence,
                n_channels,
            ).launch(
                grid=(cute.ceil_div(x.shape[0], TOKENS_PER_CTA), cute.ceil_div(x.shape[1], THREADS), 1),
                block=(THREADS, 1, 1),
                stream=stream,
            )
