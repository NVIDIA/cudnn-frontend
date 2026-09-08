# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUTLASS DSL prototype for dense and packed width-four conv backward.

This kernel is derived directly from the convolution and SiLU derivatives.  It
does not contain source from FLA, causal-conv1d, or another external kernel.

The prototype covers contiguous BF16 ``[B, T, D]`` and packed ``[1, total_T,
D]`` with device int32 ``cu_seqlens``. It supports optional BF16 channel bias,
full-width initial state, and upstream final-state gradients. A thread owns one
channel over a token tile. It streams the input once, keeps a four-element
``dz`` window in registers, and writes ``dx`` without atomics. Packed CTAs are
built on device and never cross sequence boundaries. Weight gradients have two
compile-time candidates: direct FP32 atomics, or one unique FP32 partial per
token-tile CTA followed by a second reduction kernel. Optional dbias reuses the
per-thread dz stream and emits one FP32 atomic per tile/channel in either
dweight mode. A separate one-thread-per-sequence/channel state kernel
recomputes at most the first three ``dz`` values, writes ``d_initial_state``,
and projects only the ``d_final_state`` lanes backed by initial state.
Token-backed final-state gradients join ``dx`` in the main kernel before its
single BF16 rounding. Token tile and reduction choices are a small static
schedule, not an autotuner.

Packed metadata validation is deliberately device-only. A prefix sum that
does not start at zero, does not end at ``total_T``, is not strictly
increasing, or describes an empty sequence executes a device ``trap``. This is
an asynchronous, sticky CUDA failure rather than a recoverable Python
exception; callers must discard the affected CUDA context before doing more
work.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass.cute.arch.nvvm_wrappers import inline_ptx

from cudnn.frost.tile_dsl.pointwise import sigmoid
from cudnn.sdpa.utils import atomic_add_fp32

WIDTH = 4
PACKED_TILE_DESCRIPTOR_WIDTH = 4


@cute.kernel
def _build_packed_tile_map_kernel(
    cu_seqlens: cute.Tensor,
    tile_map: cute.Tensor,
    num_sequences: cutlass.Int32,
    total_tokens: cutlass.Int64,
    tokens_per_cta: cutlass.Int32,
    tile_capacity: cutlass.Int32,
) -> None:
    """Validate packed boundaries and serialize exact per-sequence tiles.

    A descriptor is ``[sequence, start, end, tile_start]``. The capacity is the
    tight shape-only upper bound ``N + floor((total-N)/BT)``; unused entries
    are filled with empty tiles at ``total_tokens`` so the following launch
    needs no host read or dynamic grid size.
    """

    if cute.arch.block_idx()[0] == cutlass.Int32(0) and cute.arch.thread_idx()[0] == cutlass.Int32(0):
        inline_ptx(
            "trap;",
            predicate=cutlass.Int32(cu_seqlens[0]) != cutlass.Int32(0),
        )
        inline_ptx(
            "trap;",
            predicate=cutlass.Int32(cu_seqlens[num_sequences]) != total_tokens,
        )

        descriptor = cutlass.Int32(0)
        sequence = cutlass.Int32(0)
        while sequence < num_sequences:
            sequence_start = cutlass.Int64(cu_seqlens[sequence])
            sequence_end = cutlass.Int64(cu_seqlens[sequence + cutlass.Int32(1)])
            inline_ptx("trap;", predicate=sequence_start < cutlass.Int64(0))
            inline_ptx("trap;", predicate=sequence_end <= sequence_start)
            inline_ptx("trap;", predicate=sequence_end > total_tokens)

            tile_start = sequence_start
            while tile_start < sequence_end:
                inline_ptx("trap;", predicate=descriptor >= tile_capacity)
                base = cutlass.Int64(descriptor) * cutlass.Int64(PACKED_TILE_DESCRIPTOR_WIDTH)
                tile_map[base] = sequence
                tile_map[base + cutlass.Int64(1)] = cutlass.Int32(sequence_start)
                tile_map[base + cutlass.Int64(2)] = cutlass.Int32(sequence_end)
                tile_map[base + cutlass.Int64(3)] = cutlass.Int32(tile_start)
                descriptor += cutlass.Int32(1)
                tile_start += cutlass.Int64(tokens_per_cta)
            sequence += cutlass.Int32(1)

        while descriptor < tile_capacity:
            base = cutlass.Int64(descriptor) * cutlass.Int64(PACKED_TILE_DESCRIPTOR_WIDTH)
            tile_map[base] = cutlass.Int32(0)
            tile_map[base + cutlass.Int64(1)] = cutlass.Int32(total_tokens)
            tile_map[base + cutlass.Int64(2)] = cutlass.Int32(total_tokens)
            tile_map[base + cutlass.Int64(3)] = cutlass.Int32(total_tokens)
            descriptor += cutlass.Int32(1)


@cute.kernel
def _causal_conv1d_bulk_bwd_kernel(
    x: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor | None,
    dy: cute.Tensor,
    dx: cute.Tensor,
    dw_accum: cute.Tensor,
    db_accum: cute.Tensor | None,
    initial_state: cute.Tensor | None,
    d_final_state: cute.Tensor | None,
    packed_tile_map: cute.Tensor | None,
    sequence_length: cutlass.Int32,
    tokens_per_cta: cutlass.Int32,
    n_channels: cutlass.Int32,
    write_dweight_partials: cutlass.Constexpr[bool],
) -> None:
    """Fuse SiLU recompute, ``dx`` stencil, and partial ``dw`` reduction."""

    tile_linear = cutlass.Int32(cute.arch.block_idx()[0])
    channel_tile = cutlass.Int32(cute.arch.block_idx()[1])
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    threads = cutlass.Int32(cute.arch.block_dim()[0])
    channel = channel_tile * threads + tidx

    sequence_start = cutlass.Int64(0)
    sequence_end = cutlass.Int64(0)
    tile_start = cutlass.Int64(0)
    sequence = cutlass.Int32(0)
    if cutlass.const_expr(packed_tile_map is None):
        tiles_per_sequence = (sequence_length - cutlass.Int32(1)) // tokens_per_cta + cutlass.Int32(1)
        sequence = tile_linear // tiles_per_sequence
        tile_in_sequence = tile_linear - sequence * tiles_per_sequence
        sequence_start = cutlass.Int64(sequence) * cutlass.Int64(sequence_length)
        sequence_end = sequence_start + cutlass.Int64(sequence_length)
        tile_start = sequence_start + cutlass.Int64(tile_in_sequence) * cutlass.Int64(tokens_per_cta)
    else:
        descriptor = cutlass.Int64(tile_linear) * cutlass.Int64(PACKED_TILE_DESCRIPTOR_WIDTH)
        sequence = cutlass.Int32(packed_tile_map[descriptor])
        sequence_start = cutlass.Int64(packed_tile_map[descriptor + cutlass.Int64(1)])
        sequence_end = cutlass.Int64(packed_tile_map[descriptor + cutlass.Int64(2)])
        tile_start = cutlass.Int64(packed_tile_map[descriptor + cutlass.Int64(3)])
    tile_end = tile_start + cutlass.Int64(tokens_per_cta)
    if tile_end > sequence_end:  # noqa: PLR1730 -- DSL values cannot use Python min().
        tile_end = sequence_end

    w0 = cutlass.Float32(0.0)
    w1 = cutlass.Float32(0.0)
    w2 = cutlass.Float32(0.0)
    w3 = cutlass.Float32(0.0)
    bias_value = cutlass.Float32(0.0)
    if channel < n_channels:
        w0 = weight[channel, 0].to(cutlass.Float32)
        w1 = weight[channel, 1].to(cutlass.Float32)
        w2 = weight[channel, 2].to(cutlass.Float32)
        w3 = weight[channel, 3].to(cutlass.Float32)
        if cutlass.const_expr(bias is not None):
            bias_value = bias[channel].to(cutlass.Float32)

    # Input history immediately before this tile. Reads never cross a dense
    # batch-row boundary. Near a sequence start, the missing prefix comes from
    # full-width initial-state lanes 1..3; without initial state it is zero.
    h0 = cutlass.Float32(0.0)
    h1 = cutlass.Float32(0.0)
    h2 = cutlass.Float32(0.0)
    if channel < n_channels:
        local_tile_start = tile_start - sequence_start
        if tile_start - cutlass.Int64(3) >= sequence_start:
            h0 = x[tile_start - cutlass.Int64(3), channel].to(cutlass.Float32)
        elif cutlass.const_expr(initial_state is not None):
            h0 = initial_state[sequence, channel, cutlass.Int32(local_tile_start) + cutlass.Int32(1)].to(cutlass.Float32)
        if tile_start - cutlass.Int64(2) >= sequence_start:
            h1 = x[tile_start - cutlass.Int64(2), channel].to(cutlass.Float32)
        elif cutlass.const_expr(initial_state is not None):
            h1 = initial_state[sequence, channel, cutlass.Int32(local_tile_start) + cutlass.Int32(2)].to(cutlass.Float32)
        if tile_start - cutlass.Int64(1) >= sequence_start:
            h2 = x[tile_start - cutlass.Int64(1), channel].to(cutlass.Float32)
        elif cutlass.const_expr(initial_state is not None):
            h2 = initial_state[sequence, channel, cutlass.Int32(3)].to(cutlass.Float32)

    # The queue contains dz[p-3:p].  Three zero priming iterations naturally
    # delay the first dx write until all four contributing outputs are known.
    dz0 = cutlass.Float32(0.0)
    dz1 = cutlass.Float32(0.0)
    dz2 = cutlass.Float32(0.0)
    dw0 = cutlass.Float32(0.0)
    dw1 = cutlass.Float32(0.0)
    dw2 = cutlass.Float32(0.0)
    dw3 = cutlass.Float32(0.0)
    db = cutlass.Float32(0.0)

    scan_start = tile_start
    scan_end = tile_end + cutlass.Int64(3)
    p = scan_start
    while p < scan_end:
        current = cutlass.Float32(0.0)
        dz3 = cutlass.Float32(0.0)
        valid_output = p < sequence_end
        if channel < n_channels:
            if valid_output:
                current = x[p, channel].to(cutlass.Float32)
                z = h0 * w0
                z = z + h1 * w1
                z = z + h2 * w2
                z = z + current * w3
                if cutlass.const_expr(bias is not None):
                    z = z + bias_value
                gate = sigmoid(z)
                silu_grad = gate * (cutlass.Float32(1.0) + z * (cutlass.Float32(1.0) - gate))
                dz3 = dy[p, channel].to(cutlass.Float32) * silu_grad

            if p >= tile_start + cutlass.Int64(3):
                q = p - cutlass.Int64(3)
                if q < tile_end:
                    dx_value = dz0 * w3 + dz1 * w2 + dz2 * w1 + dz3 * w0
                    if cutlass.const_expr(d_final_state is not None):
                        sequence_length_local = sequence_end - sequence_start
                        final_lane = q - sequence_start + cutlass.Int64(WIDTH) - sequence_length_local
                        if final_lane >= cutlass.Int64(0):
                            if final_lane < cutlass.Int64(WIDTH):
                                dx_value = dx_value + d_final_state[sequence, channel, cutlass.Int32(final_lane)].to(cutlass.Float32)
                    dx[q, channel] = dx_value.to(cutlass.BFloat16)

            # Each token belongs to exactly one CTA's weight-gradient range.
            if p < tile_end:
                dw0 = dw0 + dz3 * h0
                dw1 = dw1 + dz3 * h1
                dw2 = dw2 + dz3 * h2
                dw3 = dw3 + dz3 * current
                if cutlass.const_expr(db_accum is not None):
                    db = db + dz3

        h0 = h1
        h1 = h2
        h2 = current
        dz0 = dz1
        dz1 = dz2
        dz2 = dz3
        p += cutlass.Int64(1)

    if channel < n_channels:
        if cutlass.const_expr(db_accum is not None):
            atomic_add_fp32(db, db_accum.iterator + channel)
        if cutlass.const_expr(write_dweight_partials):
            # tile_linear uniquely owns this [channel, tap] vector.  The
            # follow-up kernel reduces the leading partial dimension.
            partial_base = (cutlass.Int64(tile_linear) * cutlass.Int64(n_channels) + cutlass.Int64(channel)) * cutlass.Int64(WIDTH)
            partial_address = dw_accum.iterator.toint() + partial_base * cutlass.Int64(4)
            inline_ptx(
                "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
                read_only_args=[partial_address, dw0, dw1, dw2, dw3],
            )
        else:
            base = channel * cutlass.Int32(WIDTH)
            atomic_add_fp32(dw0, dw_accum.iterator + base)
            atomic_add_fp32(dw1, dw_accum.iterator + base + cutlass.Int32(1))
            atomic_add_fp32(dw2, dw_accum.iterator + base + cutlass.Int32(2))
            atomic_add_fp32(dw3, dw_accum.iterator + base + cutlass.Int32(3))


@cute.kernel
def _causal_conv1d_bulk_state_bwd_kernel(
    x: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor | None,
    dy: cute.Tensor,
    initial_state: cute.Tensor | None,
    d_final_state: cute.Tensor | None,
    d_initial_state: cute.Tensor | None,
    cu_seqlens: cute.Tensor | None,
    sequence_length: cutlass.Int32,
    num_sequences: cutlass.Int32,
    n_channels: cutlass.Int32,
) -> None:
    """Project the full-width state edge without token-tile atomics.

    One thread owns one ``[sequence, channel]`` pair. It recomputes only the
    first three output derivatives that can depend on initial state, then maps
    the upstream final-state lanes backed by initial state to their unique
    sources. Positive sequence lengths make initial-state lane zero
    unobservable, so it is always written as zero.
    """

    sequence = cutlass.Int32(cute.arch.block_idx()[0])
    channel = cutlass.Int32(cute.arch.block_idx()[1]) * cutlass.Int32(cute.arch.block_dim()[0]) + cutlass.Int32(cute.arch.thread_idx()[0])
    if sequence < num_sequences and channel < n_channels:
        sequence_start = cutlass.Int64(sequence) * cutlass.Int64(sequence_length)
        sequence_end = sequence_start + cutlass.Int64(sequence_length)
        if cutlass.const_expr(cu_seqlens is not None):
            sequence_start = cutlass.Int64(cu_seqlens[sequence])
            sequence_end = cutlass.Int64(cu_seqlens[sequence + cutlass.Int32(1)])
        length = sequence_end - sequence_start

        d_initial_1 = cutlass.Float32(0.0)
        d_initial_2 = cutlass.Float32(0.0)
        d_initial_3 = cutlass.Float32(0.0)

        if cutlass.const_expr(d_initial_state is not None):
            w0 = weight[channel, 0].to(cutlass.Float32)
            w1 = weight[channel, 1].to(cutlass.Float32)
            w2 = weight[channel, 2].to(cutlass.Float32)
            w3 = weight[channel, 3].to(cutlass.Float32)
            bias_value = cutlass.Float32(0.0)
            if cutlass.const_expr(bias is not None):
                bias_value = bias[channel].to(cutlass.Float32)

            history_0 = initial_state[sequence, channel, cutlass.Int32(1)].to(cutlass.Float32)
            history_1 = initial_state[sequence, channel, cutlass.Int32(2)].to(cutlass.Float32)
            history_2 = initial_state[sequence, channel, cutlass.Int32(3)].to(cutlass.Float32)
            state_grad_end = sequence_start + cutlass.Int64(3)
            if state_grad_end > sequence_end:  # noqa: PLR1730 -- DSL values cannot use Python min().
                state_grad_end = sequence_end

            token = sequence_start
            while token < state_grad_end:
                current = x[token, channel].to(cutlass.Float32)
                z = history_0 * w0
                z = z + history_1 * w1
                z = z + history_2 * w2
                z = z + current * w3
                if cutlass.const_expr(bias is not None):
                    z = z + bias_value
                gate = sigmoid(z)
                silu_grad = gate * (cutlass.Float32(1.0) + z * (cutlass.Float32(1.0) - gate))
                dz = dy[token, channel].to(cutlass.Float32) * silu_grad
                relative = token - sequence_start
                if relative == cutlass.Int64(0):
                    d_initial_1 = d_initial_1 + dz * w0
                    d_initial_2 = d_initial_2 + dz * w1
                    d_initial_3 = d_initial_3 + dz * w2
                elif relative == cutlass.Int64(1):
                    d_initial_2 = d_initial_2 + dz * w0
                    d_initial_3 = d_initial_3 + dz * w1
                else:
                    d_initial_3 = d_initial_3 + dz * w0
                history_0 = history_1
                history_1 = history_2
                history_2 = current
                token += cutlass.Int64(1)

        if cutlass.const_expr(d_final_state is not None):
            final_lane = cutlass.Int32(0)
            while final_lane < cutlass.Int32(WIDTH):
                grad = d_final_state[sequence, channel, final_lane].to(cutlass.Float32)
                source = length + cutlass.Int64(final_lane)
                if source < cutlass.Int64(WIDTH):
                    if cutlass.const_expr(d_initial_state is not None):
                        if source == cutlass.Int64(1):
                            d_initial_1 = d_initial_1 + grad
                        elif source == cutlass.Int64(2):
                            d_initial_2 = d_initial_2 + grad
                        else:
                            d_initial_3 = d_initial_3 + grad
                final_lane += cutlass.Int32(1)

        if cutlass.const_expr(d_initial_state is not None):
            d_initial_state[sequence, channel, cutlass.Int32(0)] = cutlass.Float32(0.0).to(cutlass.BFloat16)
            d_initial_state[sequence, channel, cutlass.Int32(1)] = d_initial_1.to(cutlass.BFloat16)
            d_initial_state[sequence, channel, cutlass.Int32(2)] = d_initial_2.to(cutlass.BFloat16)
            d_initial_state[sequence, channel, cutlass.Int32(3)] = d_initial_3.to(cutlass.BFloat16)


@cute.kernel
def _reduce_dweight_partials_kernel(
    dw_partials: cute.Tensor,
    dw_accum: cute.Tensor,
    n_partials: cutlass.Int32,
    n_channels: cutlass.Int32,
) -> None:
    """Reduce one FP32 ``[partial, channel, 4]`` vector per thread."""

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    channel = cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(cute.arch.block_dim()[0]) + tidx
    if channel < n_channels:
        dw0 = cutlass.Float32(0.0)
        dw1 = cutlass.Float32(0.0)
        dw2 = cutlass.Float32(0.0)
        dw3 = cutlass.Float32(0.0)
        partial = cutlass.Int32(0)
        while partial < n_partials:
            partial_base = (cutlass.Int64(partial) * cutlass.Int64(n_channels) + cutlass.Int64(channel)) * cutlass.Int64(WIDTH)
            partial_address = dw_partials.iterator.toint() + partial_base * cutlass.Int64(4)
            value0, value1, value2, value3 = inline_ptx(
                "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
                write_only_types=[cutlass.Float32] * WIDTH,
                read_only_args=[partial_address],
            )
            dw0 = dw0 + value0
            dw1 = dw1 + value1
            dw2 = dw2 + value2
            dw3 = dw3 + value3
            partial += cutlass.Int32(1)

        output_address = dw_accum.iterator.toint() + cutlass.Int64(channel) * cutlass.Int64(WIDTH * 4)
        inline_ptx(
            "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
            read_only_args=[output_address, dw0, dw1, dw2, dw3],
        )


class CausalConv1dBulkBackwardKernel:
    """Launch dense or per-sequence packed tiles with one dweight policy."""

    def __init__(
        self,
        *,
        batch_size: int,
        sequence_length: int,
        num_sequences: int,
        packed_tile_capacity: int,
        threads: int = 256,
        tokens_per_cta: int = 64,
        use_dweight_partials: bool = False,
        reduction_threads: int = 256,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_sequences = num_sequences
        self.packed_tile_capacity = packed_tile_capacity
        self.threads = threads
        self.tokens_per_cta = tokens_per_cta
        self.use_dweight_partials = use_dweight_partials
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
        dw_partials: cute.Tensor | None,
        cu_seqlens: cute.Tensor | None,
        initial_state: cute.Tensor | None,
        d_final_state: cute.Tensor | None,
        d_initial_state: cute.Tensor | None,
        packed_tile_map: cute.Tensor | None,
        stream: cuda.CUstream,
    ) -> None:
        tiles_per_sequence = (self.sequence_length - 1) // self.tokens_per_cta + 1
        n_partials = self.batch_size * tiles_per_sequence
        if cutlass.const_expr(cu_seqlens is not None):
            _build_packed_tile_map_kernel(
                cu_seqlens,
                packed_tile_map,
                cutlass.Int32(self.num_sequences),
                cutlass.Int64(self.batch_size * self.sequence_length),
                cutlass.Int32(self.tokens_per_cta),
                cutlass.Int32(self.packed_tile_capacity),
            ).launch(
                grid=(1, 1, 1),
                block=(1, 1, 1),
                stream=stream,
            )
            n_partials = self.packed_tile_capacity
        if cutlass.const_expr(self.use_dweight_partials):
            _causal_conv1d_bulk_bwd_kernel(
                x,
                weight,
                bias,
                dy,
                dx,
                dw_partials,
                db_accum,
                initial_state,
                d_final_state,
                packed_tile_map,
                cutlass.Int32(self.sequence_length),
                cutlass.Int32(self.tokens_per_cta),
                cutlass.Int32(x.shape[1]),
                True,
            ).launch(
                grid=(n_partials, cute.ceil_div(x.shape[1], self.threads), 1),
                block=(self.threads, 1, 1),
                stream=stream,
            )
            _reduce_dweight_partials_kernel(
                dw_partials,
                dw_accum,
                cutlass.Int32(n_partials),
                cutlass.Int32(x.shape[1]),
            ).launch(
                grid=(cute.ceil_div(x.shape[1], self.reduction_threads), 1, 1),
                block=(self.reduction_threads, 1, 1),
                stream=stream,
            )
        else:
            _causal_conv1d_bulk_bwd_kernel(
                x,
                weight,
                bias,
                dy,
                dx,
                dw_accum,
                db_accum,
                initial_state,
                d_final_state,
                packed_tile_map,
                cutlass.Int32(self.sequence_length),
                cutlass.Int32(self.tokens_per_cta),
                cutlass.Int32(x.shape[1]),
                False,
            ).launch(
                grid=(n_partials, cute.ceil_div(x.shape[1], self.threads), 1),
                block=(self.threads, 1, 1),
                stream=stream,
            )
        if cutlass.const_expr(initial_state is not None):
            _causal_conv1d_bulk_state_bwd_kernel(
                x,
                weight,
                bias,
                dy,
                initial_state,
                d_final_state,
                d_initial_state,
                cu_seqlens,
                cutlass.Int32(self.sequence_length),
                cutlass.Int32(self.num_sequences),
                cutlass.Int32(x.shape[1]),
            ).launch(
                grid=(self.num_sequences, cute.ceil_div(x.shape[1], self.threads), 1),
                block=(self.threads, 1, 1),
                stream=stream,
            )
