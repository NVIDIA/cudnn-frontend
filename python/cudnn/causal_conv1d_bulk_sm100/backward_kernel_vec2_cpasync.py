# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""B200 vec2 cp.async specialization for dense width-four conv backward.

This FE-native prototype extends the streaming backward design with a
four-stage ``cp.async`` pipeline, packed-f32x2 arithmetic, and a fast-tanh SiLU
derivative. It consumes the FE caller-owned FP32 partial workspace and reuses
the existing deterministic 16-way FP32 reducer. It does not allocate or cache
on execute and does not copy or call FLA or another external convolution
kernel.

Each 128-thread CTA owns 256 adjacent BF16 channels. Four G8 stages of X and
dY occupy 32 KiB of shared memory while each thread walks one token tile. The
main kernel writes one FP32 ``[token_tile, channel, tap]`` partial; the shared
reducer in :mod:`backward_kernel_vec4` produces the API's FP32 dW accumulator.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils
from cutlass import cute

from .backward_kernel_vec4 import _REDUCTION_SPLITS, _reduce_dweight_vec4_partials_kernel

_WIDTH = 4
_VEC = 2
_THREADS = 128
_CHANNELS_PER_CTA = _THREADS * _VEC
_TOKENS_PER_STAGE = 8
_STAGES = 4
_PACKED_PAIRS = _VEC // 2
_CHANNEL_VECTORS_16B = _CHANNELS_PER_CTA // 8
_ROWS_PER_COPY = _THREADS // _CHANNEL_VECTORS_16B
_COPIES_PER_STAGE = _TOKENS_PER_STAGE // _ROWS_PER_COPY
_SMEM_ELEMENTS = _STAGES * _TOKENS_PER_STAGE * _CHANNELS_PER_CTA
_SMEM_BYTES = 2 * _SMEM_ELEMENTS * 2
_DX_STAGE_STRIDE = (_VEC, 1)

_F32 = cutlass.Float32


@cute.jit
def _stream_step_vec2_fast_tanh(
    x_buffer,
    dy_buffer,
    weight,
    x_history,
    dz_history,
    dweight_accum,
    dx_buffer,
    dy_mask: cutlass.Float32,
    accumulate_dweight: cutlass.Constexpr,
    produce_dx: cutlass.Constexpr,
    masked: cutlass.Constexpr,
) -> None:
    """Advance one token using Blackwell packed-f32x2 arithmetic."""

    fma2 = cute.arch.fma_packed_f32x2
    mul2 = cute.arch.mul_packed_f32x2
    add2 = cute.arch.add_packed_f32x2
    one2 = (_F32(1.0), _F32(1.0))

    for pair in cutlass.range_constexpr(_PACKED_PAIRS):
        lane0 = 2 * pair
        lane1 = lane0 + 1
        current = (x_buffer[lane0].to(_F32), x_buffer[lane1].to(_F32))
        dy_value = (dy_buffer[lane0].to(_F32), dy_buffer[lane1].to(_F32))
        if cutlass.const_expr(masked):
            dy_value = mul2(dy_value, (dy_mask, dy_mask))

        # Weights are stored as w/2, so z_half is the preactivation divided by
        # two. This lets one tanh.approx evaluate the full SiLU derivative.
        z_half = mul2(
            (x_history[0, lane0], x_history[0, lane1]),
            (weight[0, lane0], weight[0, lane1]),
        )
        z_half = fma2(
            (x_history[1, lane0], x_history[1, lane1]),
            (weight[1, lane0], weight[1, lane1]),
            z_half,
        )
        z_half = fma2(
            (x_history[2, lane0], x_history[2, lane1]),
            (weight[2, lane0], weight[2, lane1]),
            z_half,
        )
        z_half = fma2(current, (weight[3, lane0], weight[3, lane1]), z_half)

        tanh_half = (
            cute.math.tanh(z_half[0], fastmath=True),
            cute.math.tanh(z_half[1], fastmath=True),
        )
        one_minus_tanh_squared = fma2((-tanh_half[0], -tanh_half[1]), tanh_half, one2)
        derivative_twice = fma2(z_half, one_minus_tanh_squared, add2(one2, tanh_half))
        dz_twice = mul2(dy_value, derivative_twice)

        if cutlass.const_expr(accumulate_dweight):
            value = fma2(
                dz_twice,
                (x_history[0, lane0], x_history[0, lane1]),
                (dweight_accum[0, lane0], dweight_accum[0, lane1]),
            )
            dweight_accum[0, lane0] = value[0]
            dweight_accum[0, lane1] = value[1]
            value = fma2(
                dz_twice,
                (x_history[1, lane0], x_history[1, lane1]),
                (dweight_accum[1, lane0], dweight_accum[1, lane1]),
            )
            dweight_accum[1, lane0] = value[0]
            dweight_accum[1, lane1] = value[1]
            value = fma2(
                dz_twice,
                (x_history[2, lane0], x_history[2, lane1]),
                (dweight_accum[2, lane0], dweight_accum[2, lane1]),
            )
            dweight_accum[2, lane0] = value[0]
            dweight_accum[2, lane1] = value[1]
            value = fma2(
                dz_twice,
                current,
                (dweight_accum[3, lane0], dweight_accum[3, lane1]),
            )
            dweight_accum[3, lane0] = value[0]
            dweight_accum[3, lane1] = value[1]

        if cutlass.const_expr(produce_dx):
            dx_value = mul2(
                (dz_history[0, lane0], dz_history[0, lane1]),
                (weight[3, lane0], weight[3, lane1]),
            )
            dx_value = fma2(
                (dz_history[1, lane0], dz_history[1, lane1]),
                (weight[2, lane0], weight[2, lane1]),
                dx_value,
            )
            dx_value = fma2(
                (dz_history[2, lane0], dz_history[2, lane1]),
                (weight[1, lane0], weight[1, lane1]),
                dx_value,
            )
            dx_value = fma2(dz_twice, (weight[0, lane0], weight[0, lane1]), dx_value)
            dx_buffer[lane0] = dx_value[0].to(cutlass.BFloat16)
            dx_buffer[lane1] = dx_value[1].to(cutlass.BFloat16)

        x_history[0, lane0] = x_history[1, lane0]
        x_history[0, lane1] = x_history[1, lane1]
        x_history[1, lane0] = x_history[2, lane0]
        x_history[1, lane1] = x_history[2, lane1]
        x_history[2, lane0] = current[0]
        x_history[2, lane1] = current[1]
        dz_history[0, lane0] = dz_history[1, lane0]
        dz_history[0, lane1] = dz_history[1, lane1]
        dz_history[1, lane0] = dz_history[2, lane0]
        dz_history[1, lane1] = dz_history[2, lane1]
        dz_history[2, lane0] = dz_twice[0]
        dz_history[2, lane1] = dz_twice[1]


@cute.jit
def _issue_cpasync_stage(
    copy_atom,
    global_x_16b,
    global_dy_16b,
    shared_x_16b,
    shared_dy_16b,
    stage: cutlass.Int32,
    token_base: cutlass.Int32,
    first_row: cutlass.Int32,
    channel_vector: cutlass.Int32,
    channel_base: cutlass.Int32,
) -> None:
    """Issue this thread's 16-byte X and dY copies for one G8 stage."""

    for copy_index in cutlass.range_constexpr(_COPIES_PER_STAGE):
        row = first_row + copy_index * _ROWS_PER_COPY
        cute.copy(
            copy_atom,
            global_x_16b[(token_base + row, channel_base + channel_vector, None)],
            shared_x_16b[(stage, row, channel_vector, None)],
        )
        cute.copy(
            copy_atom,
            global_dy_16b[(token_base + row, channel_base + channel_vector, None)],
            shared_dy_16b[(stage, row, channel_vector, None)],
        )


@cute.kernel
def _causal_conv1d_bulk_bwd_vec2_cpasync_kernel(
    x: cute.Tensor,
    weight: cute.Tensor,
    dy: cute.Tensor,
    dx: cute.Tensor,
    dweight_partials: cute.Tensor,
    sequence_length: cutlass.Int32,
    tokens_per_cta: cutlass.Int32,
    n_token_tiles: cutlass.Constexpr,
    n_channels: cutlass.Constexpr,
) -> None:
    """Run one four-stage cp.async token tile by two channels per thread."""

    thread = cutlass.Int32(cute.arch.thread_idx()[0])
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
        dweight_partials.iterator,
        cute.make_layout(
            (n_token_tiles, n_channels, _WIDTH),
            stride=(n_channels * _WIDTH, _WIDTH, 1),
        ),
    )

    packed_layout = cute.make_layout(
        (sequence_length, n_channels // 8, 8),
        stride=(n_channels, 8, 1),
    )
    # DLPack-backed tensors carry generic pointers; cp.async requires a
    # global-address-space source and the API has already checked 16B alignment.
    x_global_pointer = cute.make_ptr(
        cutlass.BFloat16,
        x.iterator.toint(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    dy_global_pointer = cute.make_ptr(
        cutlass.BFloat16,
        dy.iterator.toint(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    global_x_16b = cute.make_tensor(x_global_pointer, packed_layout)
    global_dy_16b = cute.make_tensor(dy_global_pointer, packed_layout)

    register_load = cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(),
        cutlass.BFloat16,
        num_bits_per_copy=_VEC * 16,
        invariant=True,
    )
    cpasync_load = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(
            cache_mode=cute.nvgpu.common.LoadCacheMode.GLOBAL,
        ),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )

    smem = cutlass.utils.SmemAllocator()
    shared_x_raw = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout(_SMEM_ELEMENTS),
        byte_alignment=16,
    )
    shared_dy_raw = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout(_SMEM_ELEMENTS),
        byte_alignment=16,
    )
    shared_16b_layout = cute.make_layout(
        (_STAGES, _TOKENS_PER_STAGE, _CHANNEL_VECTORS_16B, 8),
        stride=(_TOKENS_PER_STAGE * _CHANNELS_PER_CTA, _CHANNELS_PER_CTA, 8, 1),
    )
    shared_vec2_layout = cute.make_layout(
        (_STAGES, _TOKENS_PER_STAGE, _THREADS, _VEC),
        stride=(_TOKENS_PER_STAGE * _CHANNELS_PER_CTA, _CHANNELS_PER_CTA, _VEC, 1),
    )
    shared_x_16b = cute.make_tensor(shared_x_raw.iterator, shared_16b_layout)
    shared_dy_16b = cute.make_tensor(shared_dy_raw.iterator, shared_16b_layout)
    shared_x_vec2 = cute.make_tensor(shared_x_raw.iterator, shared_vec2_layout)
    shared_dy_vec2 = cute.make_tensor(shared_dy_raw.iterator, shared_vec2_layout)

    channel_vector = channel_tile * (_CHANNELS_PER_CTA // _VEC) + thread
    channel_base = channel_tile * _CHANNEL_VECTORS_16B
    first_row = thread // _CHANNEL_VECTORS_16B
    channel_vector_16b = thread % _CHANNEL_VECTORS_16B

    token_start = token_tile * tokens_per_cta
    token_end = token_start + tokens_per_cta
    if token_end > sequence_length:
        token_end = sequence_length
    last_full_stage_start = sequence_length - _TOKENS_PER_STAGE

    x_history = cute.make_rmem_tensor((3, _VEC), _F32)
    dz_history = cute.make_rmem_tensor((3, _VEC), _F32)
    dweight_accum = cute.make_rmem_tensor((_WIDTH, _VEC), _F32)
    weight_values = cute.make_rmem_tensor((_WIDTH, _VEC), _F32)
    x_buffer = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
    dy_buffer = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
    dx_buffer = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
    dx_stage = cute.make_rmem_tensor(
        cute.make_layout(
            (_TOKENS_PER_STAGE, _VEC),
            stride=_DX_STAGE_STRIDE,
        ),
        cutlass.BFloat16,
    )

    for history in cutlass.range_constexpr(3):
        for lane in cutlass.range_constexpr(_VEC):
            x_history[history, lane] = _F32(0.0)
            dz_history[history, lane] = _F32(0.0)
    for tap in cutlass.range_constexpr(_WIDTH):
        for lane in cutlass.range_constexpr(_VEC):
            dweight_accum[tap, lane] = _F32(0.0)

    packed_weight = cute.make_rmem_tensor((_WIDTH * _VEC,), weight.element_type)
    cute.autovec_copy(global_weight[(channel_vector, None)], packed_weight)
    for lane in cutlass.range_constexpr(_VEC):
        for tap in cutlass.range_constexpr(_WIDTH):
            weight_values[tap, lane] = packed_weight[_WIDTH * lane + tap].to(_F32) * _F32(0.5)

    staged_start = token_start + 3
    staged_tokens = token_end - staged_start
    full_stages = staged_tokens // _TOKENS_PER_STAGE

    # Fill three stages before the serial three-token prime.
    for stage in cutlass.range_constexpr(_STAGES - 1):
        if stage < full_stages:
            issue_start = staged_start + stage * _TOKENS_PER_STAGE
            if issue_start > last_full_stage_start:
                issue_start = last_full_stage_start
            _issue_cpasync_stage(
                cpasync_load,
                global_x_16b,
                global_dy_16b,
                shared_x_16b,
                shared_dy_16b,
                stage,
                issue_start,
                first_row,
                channel_vector_16b,
                channel_base,
            )
        cute.arch.cp_async_commit_group()

    # Coalesce the three halo and three prime-token register loads before any
    # arithmetic to preserve the schedule's latency-hiding structure.
    prologue_layout = cute.make_layout((3, _VEC), stride=(_VEC, 1))
    halo_x = cute.make_rmem_tensor(prologue_layout, cutlass.BFloat16)
    prime_x = cute.make_rmem_tensor(prologue_layout, cutlass.BFloat16)
    prime_dy = cute.make_rmem_tensor(prologue_layout, cutlass.BFloat16)
    halo_start = token_start - 3
    if halo_start < 0:
        halo_start = cutlass.Int32(0)
    for index in cutlass.range_constexpr(3):
        cute.copy(register_load, global_x[(halo_start + index, channel_vector, None)], halo_x[(index, None)])
        cute.copy(register_load, global_x[(token_start + index, channel_vector, None)], prime_x[(index, None)])
        cute.copy(register_load, global_dy[(token_start + index, channel_vector, None)], prime_dy[(index, None)])

    if token_start >= 3:
        for index in cutlass.range_constexpr(3):
            for lane in cutlass.range_constexpr(_VEC):
                x_history[index, lane] = halo_x[index, lane].to(_F32)

    one = _F32(1.0)
    for index in cutlass.range_constexpr(3):
        _stream_step_vec2_fast_tanh(
            prime_x[(index, None)],
            prime_dy[(index, None)],
            weight_values,
            x_history,
            dz_history,
            dweight_accum,
            dx_buffer,
            one,
            True,
            False,
            False,
        )

    stage = cutlass.Int32(0)
    next_stage = cutlass.Int32(_STAGES - 1)
    for stage_index in cutlass.range(full_stages, unroll=1):
        cute.arch.cp_async_wait_group(_STAGES - 2)
        cute.arch.barrier()
        stage_start = staged_start + stage_index * _TOKENS_PER_STAGE
        for token in cutlass.range_constexpr(_TOKENS_PER_STAGE):
            cute.autovec_copy(shared_x_vec2[(stage, token, thread, None)], x_buffer)
            cute.autovec_copy(shared_dy_vec2[(stage, token, thread, None)], dy_buffer)
            _stream_step_vec2_fast_tanh(
                x_buffer,
                dy_buffer,
                weight_values,
                x_history,
                dz_history,
                dweight_accum,
                dx_buffer,
                one,
                True,
                True,
                False,
            )
            cute.autovec_copy(dx_buffer, dx_stage[(token, None)])

        # This stage was last read one iteration ago; the top-of-loop barrier
        # already separates that read from the following refill.
        if stage_index + (_STAGES - 1) < full_stages:
            issue_start = stage_start + (_STAGES - 1) * _TOKENS_PER_STAGE
            if issue_start > last_full_stage_start:
                issue_start = last_full_stage_start
            _issue_cpasync_stage(
                cpasync_load,
                global_x_16b,
                global_dy_16b,
                shared_x_16b,
                shared_dy_16b,
                next_stage,
                issue_start,
                first_row,
                channel_vector_16b,
                channel_base,
            )
        cute.arch.cp_async_commit_group()
        # Keep arithmetic and token ownership unchanged, but issue each G8
        # stage's adjacent dX stores together after the next cp.async refill.
        for token in cutlass.range_constexpr(_TOKENS_PER_STAGE):
            cute.autovec_copy(
                dx_stage[(token, None)],
                global_dx[(stage_start + token - 3, channel_vector, None)],
            )
        stage = stage + 1
        if stage == _STAGES:
            stage = cutlass.Int32(0)
        next_stage = next_stage + 1
        if next_stage == _STAGES:
            next_stage = cutlass.Int32(0)

    tail_start = staged_start + full_stages * _TOKENS_PER_STAGE
    for index in cutlass.range(staged_tokens - full_stages * _TOKENS_PER_STAGE, unroll=1):
        cute.copy(register_load, global_x[(tail_start + index, channel_vector, None)], x_buffer)
        cute.copy(register_load, global_dy[(tail_start + index, channel_vector, None)], dy_buffer)
        _stream_step_vec2_fast_tanh(
            x_buffer,
            dy_buffer,
            weight_values,
            x_history,
            dz_history,
            dweight_accum,
            dx_buffer,
            one,
            True,
            True,
            False,
        )
        cute.autovec_copy(dx_buffer, global_dx[(tail_start + index - 3, channel_vector, None)])

    # Drain dz[token_end:token_end+3], masking the sequence boundary. These
    # values finish this tile's final three dX values but never contribute dW.
    for index in cutlass.range_constexpr(3):
        source_token = token_end + index
        mask = _F32(0.0)
        clamped_token = sequence_length - 1
        if source_token < sequence_length:
            mask = _F32(1.0)
            clamped_token = source_token
        cute.copy(register_load, global_x[(clamped_token, channel_vector, None)], x_buffer)
        cute.copy(register_load, global_dy[(clamped_token, channel_vector, None)], dy_buffer)
        _stream_step_vec2_fast_tanh(
            x_buffer,
            dy_buffer,
            weight_values,
            x_history,
            dz_history,
            dweight_accum,
            dx_buffer,
            mask,
            False,
            True,
            True,
        )
        cute.autovec_copy(dx_buffer, global_dx[(token_end + index - 3, channel_vector, None)])

    # dz and weights were each scaled by two in opposite directions. Apply the
    # remaining 0.5 once while materializing the exact FP32 dW partial.
    channel_partial = cute.make_rmem_tensor((_WIDTH,), _F32)
    for lane in cutlass.range_constexpr(_VEC):
        for tap in cutlass.range_constexpr(_WIDTH):
            channel_partial[tap] = dweight_accum[tap, lane] * _F32(0.5)
        cute.autovec_copy(
            channel_partial,
            global_partials[(token_tile, channel_vector * _VEC + lane, None)],
        )


class CausalConv1dBulkBackwardVec2CpAsyncKernel:
    """Launch the B200 cp.async main kernel and existing FP32 reducer."""

    def __init__(
        self,
        *,
        sequence_length: int,
        n_channels: int,
        tokens_per_cta: int,
        n_token_tiles: int,
        reduction_threads: int,
    ) -> None:
        if sequence_length < 3:
            raise ValueError(f"sequence_length must be at least 3, got {sequence_length}")
        if n_channels % _CHANNELS_PER_CTA != 0:
            raise ValueError(f"n_channels must be divisible by {_CHANNELS_PER_CTA}, got {n_channels}")
        if tokens_per_cta < 3 or (tokens_per_cta - 3) % _TOKENS_PER_STAGE != 0:
            raise ValueError(f"tokens_per_cta must be at least 3 and congruent to 3 modulo {_TOKENS_PER_STAGE}, got {tokens_per_cta}")
        expected_tiles = (sequence_length + tokens_per_cta - 1) // tokens_per_cta
        if n_token_tiles != expected_tiles:
            raise ValueError(f"n_token_tiles mismatch: expected {expected_tiles}, got {n_token_tiles}")
        last_tile_tokens = sequence_length - (n_token_tiles - 1) * tokens_per_cta
        if last_tile_tokens < 3:
            raise ValueError(f"the final token tile must contain at least 3 tokens, got {last_tile_tokens}")
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
        dweight_partials: cute.Tensor,
        cu_seqlens: cute.Tensor | None,
        initial_state: cute.Tensor | None,
        d_final_state: cute.Tensor | None,
        d_initial_state: cute.Tensor | None,
        packed_tile_map: cute.Tensor | None,
        stream: cuda.CUstream,
    ) -> None:
        _causal_conv1d_bulk_bwd_vec2_cpasync_kernel(
            x,
            weight,
            dy,
            dx,
            dweight_partials,
            cutlass.Int32(self.sequence_length),
            cutlass.Int32(self.tokens_per_cta),
            self.n_token_tiles,
            self.n_channels,
        ).launch(
            grid=(self.n_channels // _CHANNELS_PER_CTA, self.n_token_tiles, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
            smem=_SMEM_BYTES,
        )
        _reduce_dweight_vec4_partials_kernel(
            dweight_partials,
            dw_accum,
            cutlass.Int32(self.n_token_tiles),
            self.n_channels,
            self.reduction_threads // _REDUCTION_SPLITS,
        ).launch(
            grid=(self.n_channels // (self.reduction_threads // _REDUCTION_SPLITS), 1, 1),
            block=(self.reduction_threads // _REDUCTION_SPLITS, _REDUCTION_SPLITS, 1),
            stream=stream,
        )


__all__ = ["CausalConv1dBulkBackwardVec2CpAsyncKernel"]
