# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 causal-convolution decode update kernel.

The state transition follows the public ``causal_conv1d_update`` contract used
by FLA's ``ShortConvolution.step``: append the current token to the history,
retain the configured state length, take a four-wide depthwise dot product,
then optionally apply SiLU. No source from either behavioral reference below
is included. This implementation uses CUTLASS/CuTe DSL, inline PTX, and in-tree
NVIDIA FROST primitives.

Semantic references consulted:

* fla-org/flash-linear-attention, ``fla/modules/conv/short_conv.py`` and
  ``fla/modules/conv/triton/kernels.py`` (MIT), FLA 0.5.2.
* Dao-AILab/causal-conv1d's public ``causal_conv1d_update`` API contract
  (BSD-3-Clause), revision ``cd81f0413cad2fc1e6f17e785ac39f59aae690cd``.

Only the four-wide, BF16, optional-bias inference specialization lives here.
State length four keeps the original vectorized path. State length three, the
standard ``W - 1`` prefill handoff, uses a separate scalar state specialization.
Identity and SiLU are separate compile-time specializations.
One decode row is assigned to each CTA on every admitted architecture. This is
an independent FE-native implementation using the original inline-PTX data
path.
"""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.arch.nvvm_wrappers import inline_ptx

from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, sigmoid

THREADS = 256


@cute.kernel
def _causal_conv1d_update_kernel(
    x: cute.Tensor,
    weight: cute.Tensor,
    bias: Optional[cute.Tensor],
    state: cute.Tensor,
    output: cute.Tensor,
    state_indices: Optional[cute.Tensor],
    n_slots: cutlass.Int32,
    n_channels: cutlass.Int32,
    apply_silu: cutlass.Constexpr[bool],
) -> None:
    """Advance one 256-channel tile of a state row and emit its output tile.

    ``state_indices`` is optional-specialized.  ``-1`` is a padding row: it
    writes a zero output and does not mutate state.  Other indexed calls trap
    on an out-of-range or duplicate slot before that CTA writes state.
    Rejecting duplicates makes the mutation deterministic rather than allowing
    two decode rows to race on the same mutable cache slot.
    """

    row = cutlass.Int32(cute.arch.block_idx()[0])
    channel_tile = cutlass.Int32(cute.arch.block_idx()[1])
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])

    slot = row
    if cutlass.const_expr(state_indices is not None):
        slot = cutlass.Int32(state_indices[row])
        if tidx == cutlass.Int32(0):
            # CuTe's testing assert lowers away in release device builds.  PTX
            # trap is the fail-closed primitive here and is covered by GPU
            # subprocess tests for every invalid-index class.
            inline_ptx("trap;", predicate=slot < cutlass.Int32(-1))
            inline_ptx("trap;", predicate=slot >= n_slots)

            # Decode batches are small.  A lane-zero scan keeps the operation
            # single-kernel while failing closed on duplicate mutable slots.
            # Padding rows do not own state, so repeated -1 entries are valid.
            if slot >= cutlass.Int32(0):
                previous_row = cutlass.Int32(0)
                while previous_row < row:
                    inline_ptx(
                        "trap;",
                        predicate=cutlass.Int32(state_indices[previous_row]) == slot,
                    )
                    previous_row += cutlass.Int32(1)
        cute.arch.barrier()

    if cutlass.const_expr(state_indices is not None):
        is_padding = slot == cutlass.Int32(-1)
        if is_padding:
            # Keep all later pointer arithmetic in range.  Loads from slot zero
            # are harmless because the indexed padding branch suppresses the
            # state store and observable convolution result.
            slot = cutlass.Int32(0)

    channel = channel_tile * cutlass.Int32(THREADS) + tidx
    if channel < n_channels:
        state_element = cutlass.Int64(slot) * cutlass.Int64(state.stride[0]) + cutlass.Int64(channel) * cutlass.Int64(state.stride[1])
        state_address = state.iterator.toint() + state_element * cutlass.Int64(2)
        weight_address = weight.iterator.toint() + cutlass.Int64(channel) * cutlass.Int64(weight.stride[0]) * cutlass.Int64(2)
        x_element = cutlass.Int64(row) * cutlass.Int64(x.stride[0]) + cutlass.Int64(channel) * cutlass.Int64(x.stride[1])
        x_address = x.iterator.toint() + x_element * cutlass.Int64(2)

        state_01, state_23 = inline_ptx(
            "ld.global.v2.b32 {$0, $1}, [$2];",
            write_only_types=[cutlass.Int32, cutlass.Int32],
            read_only_args=[state_address],
        )
        weight_01, weight_23 = inline_ptx(
            "ld.global.v2.b32 {$0, $1}, [$2];",
            write_only_types=[cutlass.Int32, cutlass.Int32],
            read_only_args=[weight_address],
        )
        x_bits = inline_ptx(
            "ld.global.u16 $0, [$1];",
            write_only_types=[cutlass.Uint16],
            read_only_args=[x_address],
        )

        _state_0, state_1 = f16x2_to_f32(state_01, dtype=cutlass.BFloat16)
        state_2, state_3 = f16x2_to_f32(state_23, dtype=cutlass.BFloat16)
        weight_0, weight_1 = f16x2_to_f32(weight_01, dtype=cutlass.BFloat16)
        weight_2, weight_3 = f16x2_to_f32(weight_23, dtype=cutlass.BFloat16)
        bias_value = cutlass.Float32(0.0)
        if cutlass.const_expr(bias is not None):
            bias_value = bias[channel].to(cutlass.Float32)
        x_f32 = inline_ptx(
            "{ .reg .b16 x; mov.b16 x, $1; mov.b32 $0, {0, x}; }",
            write_only_types=[cutlass.Float32],
            read_only_args=[x_bits],
        )

        # The updated state is [old_1, old_2, old_3, x].  Pack from the raw
        # BF16 words so the shift and append are bitwise, including signed zero
        # and NaN payloads; only the output path converts through FP32.
        updated_01, updated_23 = inline_ptx(
            "{ .reg .b16 s0, s1, s2, s3, xv; " "mov.b32 {s0, s1}, $2; mov.b32 {s2, s3}, $3; mov.b16 xv, $4; " "mov.b32 $0, {s1, s2}; mov.b32 $1, {s3, xv}; }",
            write_only_types=[cutlass.Int32, cutlass.Int32],
            read_only_args=[state_01, state_23, x_bits],
        )
        if cutlass.const_expr(state_indices is not None):
            if not is_padding:
                inline_ptx(
                    "st.global.v2.b32 [$0], {$1, $2};",
                    read_only_args=[state_address, updated_01, updated_23],
                )
        else:
            inline_ptx(
                "st.global.v2.b32 [$0], {$1, $2};",
                read_only_args=[state_address, updated_01, updated_23],
            )

        acc = state_1 * weight_0
        acc = acc + state_2 * weight_1
        acc = acc + state_3 * weight_2
        acc = acc + x_f32 * weight_3
        acc = acc + bias_value
        if cutlass.const_expr(apply_silu):
            acc = acc * sigmoid(acc)

        if cutlass.const_expr(state_indices is not None):
            if is_padding:
                output[row, channel] = cutlass.Float32(0.0).to(cutlass.BFloat16)
            else:
                output[row, channel] = acc.to(cutlass.BFloat16)
        else:
            output[row, channel] = acc.to(cutlass.BFloat16)


@cute.kernel
def _causal_conv1d_update_l3_kernel(
    x: cute.Tensor,
    weight: cute.Tensor,
    bias: Optional[cute.Tensor],
    state: cute.Tensor,
    output: cute.Tensor,
    state_indices: Optional[cute.Tensor],
    n_slots: cutlass.Int32,
    n_channels: cutlass.Int32,
    apply_silu: cutlass.Constexpr[bool],
) -> None:
    """Advance a three-element ``W - 1`` state for one decode token.

    This specialization deliberately uses scalar state loads and stores. The
    four-element kernel above retains its original vector load/store path.
    """

    row = cutlass.Int32(cute.arch.block_idx()[0])
    channel_tile = cutlass.Int32(cute.arch.block_idx()[1])
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])

    slot = row
    if cutlass.const_expr(state_indices is not None):
        slot = cutlass.Int32(state_indices[row])
        if tidx == cutlass.Int32(0):
            inline_ptx("trap;", predicate=slot < cutlass.Int32(-1))
            inline_ptx("trap;", predicate=slot >= n_slots)
            if slot >= cutlass.Int32(0):
                previous_row = cutlass.Int32(0)
                while previous_row < row:
                    inline_ptx(
                        "trap;",
                        predicate=cutlass.Int32(state_indices[previous_row]) == slot,
                    )
                    previous_row += cutlass.Int32(1)
        cute.arch.barrier()

    if cutlass.const_expr(state_indices is not None):
        is_padding = slot == cutlass.Int32(-1)
        if is_padding:
            slot = cutlass.Int32(0)

    channel = channel_tile * cutlass.Int32(THREADS) + tidx
    if channel < n_channels:
        state_element = cutlass.Int64(slot) * cutlass.Int64(state.stride[0]) + cutlass.Int64(channel) * cutlass.Int64(state.stride[1])
        state_address = state.iterator.toint() + state_element * cutlass.Int64(2)
        state_1_address = state_address + cutlass.Int64(2)
        state_2_address = state_address + cutlass.Int64(4)
        weight_address = weight.iterator.toint() + cutlass.Int64(channel) * cutlass.Int64(weight.stride[0]) * cutlass.Int64(2)
        x_element = cutlass.Int64(row) * cutlass.Int64(x.stride[0]) + cutlass.Int64(channel) * cutlass.Int64(x.stride[1])
        x_address = x.iterator.toint() + x_element * cutlass.Int64(2)

        state_0_bits = inline_ptx(
            "ld.global.u16 $0, [$1];",
            write_only_types=[cutlass.Uint16],
            read_only_args=[state_address],
        )
        state_1_bits = inline_ptx(
            "ld.global.u16 $0, [$1];",
            write_only_types=[cutlass.Uint16],
            read_only_args=[state_1_address],
        )
        state_2_bits = inline_ptx(
            "ld.global.u16 $0, [$1];",
            write_only_types=[cutlass.Uint16],
            read_only_args=[state_2_address],
        )
        weight_01, weight_23 = inline_ptx(
            "ld.global.v2.b32 {$0, $1}, [$2];",
            write_only_types=[cutlass.Int32, cutlass.Int32],
            read_only_args=[weight_address],
        )
        x_bits = inline_ptx(
            "ld.global.u16 $0, [$1];",
            write_only_types=[cutlass.Uint16],
            read_only_args=[x_address],
        )

        state_0 = inline_ptx(
            "{ .reg .b16 x; mov.b16 x, $1; mov.b32 $0, {0, x}; }",
            write_only_types=[cutlass.Float32],
            read_only_args=[state_0_bits],
        )
        state_1 = inline_ptx(
            "{ .reg .b16 x; mov.b16 x, $1; mov.b32 $0, {0, x}; }",
            write_only_types=[cutlass.Float32],
            read_only_args=[state_1_bits],
        )
        state_2 = inline_ptx(
            "{ .reg .b16 x; mov.b16 x, $1; mov.b32 $0, {0, x}; }",
            write_only_types=[cutlass.Float32],
            read_only_args=[state_2_bits],
        )
        weight_0, weight_1 = f16x2_to_f32(weight_01, dtype=cutlass.BFloat16)
        weight_2, weight_3 = f16x2_to_f32(weight_23, dtype=cutlass.BFloat16)
        bias_value = cutlass.Float32(0.0)
        if cutlass.const_expr(bias is not None):
            bias_value = bias[channel].to(cutlass.Float32)
        x_f32 = inline_ptx(
            "{ .reg .b16 x; mov.b16 x, $1; mov.b32 $0, {0, x}; }",
            write_only_types=[cutlass.Float32],
            read_only_args=[x_bits],
        )

        # Preserve BF16 payload bits in the state transition [old_1, old_2, x].
        # A three-element channel row is only two-byte aligned for odd channels,
        # so every state access is deliberately scalar.
        if cutlass.const_expr(state_indices is not None):
            if not is_padding:
                inline_ptx(
                    "st.global.u16 [$0], $1;",
                    read_only_args=[state_address, state_1_bits],
                )
                inline_ptx(
                    "st.global.u16 [$0], $1;",
                    read_only_args=[state_1_address, state_2_bits],
                )
                inline_ptx(
                    "st.global.u16 [$0], $1;",
                    read_only_args=[state_2_address, x_bits],
                )
        else:
            inline_ptx(
                "st.global.u16 [$0], $1;",
                read_only_args=[state_address, state_1_bits],
            )
            inline_ptx(
                "st.global.u16 [$0], $1;",
                read_only_args=[state_1_address, state_2_bits],
            )
            inline_ptx(
                "st.global.u16 [$0], $1;",
                read_only_args=[state_2_address, x_bits],
            )

        acc = state_0 * weight_0
        acc = acc + state_1 * weight_1
        acc = acc + state_2 * weight_2
        acc = acc + x_f32 * weight_3
        acc = acc + bias_value
        if cutlass.const_expr(apply_silu):
            acc = acc * sigmoid(acc)

        if cutlass.const_expr(state_indices is not None):
            if is_padding:
                output[row, channel] = cutlass.Float32(0.0).to(cutlass.BFloat16)
            else:
                output[row, channel] = acc.to(cutlass.BFloat16)
        else:
            output[row, channel] = acc.to(cutlass.BFloat16)


class _CausalConv1dUpdateKernel:
    """Host launcher for the portable one-row decode specialization."""

    def __init__(self, *, apply_silu: bool, state_len: int):
        self.apply_silu = apply_silu
        self.state_len = state_len

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        bias: Optional[cute.Tensor],
        state: cute.Tensor,
        output: cute.Tensor,
        state_indices: Optional[cute.Tensor],
        n_slots: cutlass.Int32,
        n_channels: cutlass.Int32,
        stream: cuda.CUstream,
    ) -> None:
        if cutlass.const_expr(self.state_len == 4):
            _causal_conv1d_update_kernel(
                x,
                weight,
                bias,
                state,
                output,
                state_indices,
                n_slots,
                n_channels,
                self.apply_silu,
            ).launch(
                grid=(x.shape[0], cute.ceil_div(x.shape[1], THREADS), 1),
                block=(THREADS, 1, 1),
                stream=stream,
            )
        else:
            _causal_conv1d_update_l3_kernel(
                x,
                weight,
                bias,
                state,
                output,
                state_indices,
                n_slots,
                n_channels,
                self.apply_silu,
            ).launch(
                grid=(x.shape[0], cute.ceil_div(x.shape[1], THREADS), 1),
                block=(THREADS, 1, 1),
                stream=stream,
            )
