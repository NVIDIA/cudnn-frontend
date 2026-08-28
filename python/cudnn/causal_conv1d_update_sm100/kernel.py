# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 BF16 causal-convolution decode update kernel.

The state transition follows the public ``causal_conv1d_update`` contract used
by FLA's ``ShortConvolution.step``: shift the four-element history left, append
the current token, take a depthwise dot product, then apply SiLU.  No source
from either behavioral reference below is included.  This implementation uses
CUTLASS/CuTe DSL, inline PTX, and in-tree NVIDIA FROST primitives.

Semantic references consulted:

* fla-org/flash-linear-attention, ``fla/modules/conv/short_conv.py`` and
  ``fla/modules/conv/triton/kernels.py`` (MIT), FLA 0.5.2.
* Dao-AILab/causal-conv1d's public ``causal_conv1d_update`` API contract
  (BSD-3-Clause), revision ``cd81f0413cad2fc1e6f17e785ac39f59aae690cd``.

Only the four-wide, BF16, no-bias, SiLU inference specialization lives here.
The general and indexed kernel assigns one decode row to each CTA.  A narrow
``N=128`` no-index specialization assigns two rows to a CTA and reuses each
channel's weight vector across them.  That scheduling idea came from audited
internal Kernel Factory candidate
``73d90c7fa5ae2e3e2ceb195bfa5af4db87cff8aaaefb7938cdd96b3128dd3a2e``;
the standalone candidate source is not included here.  This is an independent
FE-native implementation which retains the original inline-PTX data path.
"""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.arch.nvvm_wrappers import inline_ptx

from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, sigmoid

THREADS = 256
ROW_BATCH_ROWS = 2
ROW_BATCH_SHAPES = frozenset(((128, 2048), (128, 4096)))


def select_rows_per_cta(
    n_rows: int,
    n_channels: int,
    has_state_indices: bool,
) -> int:
    """Select the audited two-row schedule only for its measured domain.

    Indexed calls deliberately retain one-row CTAs: their per-row range and
    duplicate validation is part of the mutation contract, and the Kernel
    Factory artifact did not implement that ABI.
    """

    if not has_state_indices and (n_rows, n_channels) in ROW_BATCH_SHAPES:
        return ROW_BATCH_ROWS
    return 1


@cute.kernel
def _causal_conv1d_update_kernel(
    x: cute.Tensor,
    weight: cute.Tensor,
    state: cute.Tensor,
    output: cute.Tensor,
    state_indices: Optional[cute.Tensor],
    n_slots: cutlass.Int32,
    n_channels: cutlass.Int32,
) -> None:
    """Advance one 256-channel tile of a state row and emit its output tile.

    ``state_indices`` is optional-specialized.  Indexed calls trap on an
    out-of-range or duplicate slot before that CTA writes state.
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
            # trap is the fail-closed primitive here and is covered by exact
            # SM100 subprocess tests for every invalid-index class.
            inline_ptx("trap;", predicate=slot < cutlass.Int32(0))
            inline_ptx("trap;", predicate=slot >= n_slots)

            # Decode batches are small.  A lane-zero scan keeps the operation
            # single-kernel while failing closed on duplicate mutable slots.
            previous_row = cutlass.Int32(0)
            while previous_row < row:
                inline_ptx(
                    "trap;",
                    predicate=cutlass.Int32(state_indices[previous_row]) == slot,
                )
                previous_row += cutlass.Int32(1)
        cute.arch.barrier()

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
        inline_ptx(
            "st.global.v2.b32 [$0], {$1, $2};",
            read_only_args=[state_address, updated_01, updated_23],
        )

        acc = state_1 * weight_0
        acc = acc + state_2 * weight_1
        acc = acc + state_3 * weight_2
        acc = acc + x_f32 * weight_3
        output[row, channel] = (acc * sigmoid(acc)).to(cutlass.BFloat16)


class CausalConv1dUpdateKernel:
    """Host launcher for the fixed SM100 decode specialization."""

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        state: cute.Tensor,
        output: cute.Tensor,
        state_indices: Optional[cute.Tensor],
        n_slots: cutlass.Int32,
        n_channels: cutlass.Int32,
        stream: cuda.CUstream,
    ) -> None:
        _causal_conv1d_update_kernel(
            x,
            weight,
            state,
            output,
            state_indices,
            n_slots,
            n_channels,
        ).launch(
            grid=(x.shape[0], cute.ceil_div(x.shape[1], THREADS), 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )


@cute.kernel
def _causal_conv1d_update_row_batch_kernel(
    x: cute.Tensor,
    weight: cute.Tensor,
    state: cute.Tensor,
    output: cute.Tensor,
    n_channels: cutlass.Int32,
) -> None:
    """Advance two unindexed decode rows while loading each weight once.

    The host only selects this kernel for ``N=128`` and ``D`` in
    ``{2048, 4096}``.  Those dimensions are divisible by the 256-thread channel
    tile, so every CTA owns exactly two complete, disjoint row/channel tiles.
    Both rows' state and input loads are issued before either row's arithmetic.
    """

    row_0 = cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(ROW_BATCH_ROWS)
    row_1 = row_0 + cutlass.Int32(1)
    channel_tile = cutlass.Int32(cute.arch.block_idx()[1])
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    channel = channel_tile * cutlass.Int32(THREADS) + tidx

    if channel < n_channels:
        weight_address = weight.iterator.toint() + cutlass.Int64(channel) * cutlass.Int64(weight.stride[0]) * cutlass.Int64(2)
        state_element_0 = cutlass.Int64(row_0) * cutlass.Int64(state.stride[0]) + cutlass.Int64(channel) * cutlass.Int64(state.stride[1])
        state_element_1 = cutlass.Int64(row_1) * cutlass.Int64(state.stride[0]) + cutlass.Int64(channel) * cutlass.Int64(state.stride[1])
        state_address_0 = state.iterator.toint() + state_element_0 * cutlass.Int64(2)
        state_address_1 = state.iterator.toint() + state_element_1 * cutlass.Int64(2)
        x_element_0 = cutlass.Int64(row_0) * cutlass.Int64(x.stride[0]) + cutlass.Int64(channel) * cutlass.Int64(x.stride[1])
        x_element_1 = cutlass.Int64(row_1) * cutlass.Int64(x.stride[0]) + cutlass.Int64(channel) * cutlass.Int64(x.stride[1])
        x_address_0 = x.iterator.toint() + x_element_0 * cutlass.Int64(2)
        x_address_1 = x.iterator.toint() + x_element_1 * cutlass.Int64(2)

        # Reuse the four-tap channel weight for both rows.  Keep both rows'
        # independent memory requests in flight before consuming either one.
        weight_01, weight_23 = inline_ptx(
            "ld.global.v2.b32 {$0, $1}, [$2];",
            write_only_types=[cutlass.Int32, cutlass.Int32],
            read_only_args=[weight_address],
        )
        state_0_01, state_0_23 = inline_ptx(
            "ld.global.v2.b32 {$0, $1}, [$2];",
            write_only_types=[cutlass.Int32, cutlass.Int32],
            read_only_args=[state_address_0],
        )
        x_bits_0 = inline_ptx(
            "ld.global.u16 $0, [$1];",
            write_only_types=[cutlass.Uint16],
            read_only_args=[x_address_0],
        )
        state_1_01, state_1_23 = inline_ptx(
            "ld.global.v2.b32 {$0, $1}, [$2];",
            write_only_types=[cutlass.Int32, cutlass.Int32],
            read_only_args=[state_address_1],
        )
        x_bits_1 = inline_ptx(
            "ld.global.u16 $0, [$1];",
            write_only_types=[cutlass.Uint16],
            read_only_args=[x_address_1],
        )

        weight_0, weight_1 = f16x2_to_f32(weight_01, dtype=cutlass.BFloat16)
        weight_2, weight_3 = f16x2_to_f32(weight_23, dtype=cutlass.BFloat16)

        _state_0_0, state_0_1 = f16x2_to_f32(state_0_01, dtype=cutlass.BFloat16)
        state_0_2, state_0_3 = f16x2_to_f32(state_0_23, dtype=cutlass.BFloat16)
        x_f32_0 = inline_ptx(
            "{ .reg .b16 x; mov.b16 x, $1; mov.b32 $0, {0, x}; }",
            write_only_types=[cutlass.Float32],
            read_only_args=[x_bits_0],
        )

        _state_1_0, state_1_1 = f16x2_to_f32(state_1_01, dtype=cutlass.BFloat16)
        state_1_2, state_1_3 = f16x2_to_f32(state_1_23, dtype=cutlass.BFloat16)
        x_f32_1 = inline_ptx(
            "{ .reg .b16 x; mov.b16 x, $1; mov.b32 $0, {0, x}; }",
            write_only_types=[cutlass.Float32],
            read_only_args=[x_bits_1],
        )

        # Pack the shift from the original BF16 payloads.  This retains exact
        # state bits, including signed zero and NaN payloads, for both rows.
        updated_0_01, updated_0_23 = inline_ptx(
            "{ .reg .b16 s0, s1, s2, s3, xv; " "mov.b32 {s0, s1}, $2; mov.b32 {s2, s3}, $3; mov.b16 xv, $4; " "mov.b32 $0, {s1, s2}; mov.b32 $1, {s3, xv}; }",
            write_only_types=[cutlass.Int32, cutlass.Int32],
            read_only_args=[state_0_01, state_0_23, x_bits_0],
        )
        updated_1_01, updated_1_23 = inline_ptx(
            "{ .reg .b16 s0, s1, s2, s3, xv; " "mov.b32 {s0, s1}, $2; mov.b32 {s2, s3}, $3; mov.b16 xv, $4; " "mov.b32 $0, {s1, s2}; mov.b32 $1, {s3, xv}; }",
            write_only_types=[cutlass.Int32, cutlass.Int32],
            read_only_args=[state_1_01, state_1_23, x_bits_1],
        )
        inline_ptx(
            "st.global.v2.b32 [$0], {$1, $2};",
            read_only_args=[state_address_0, updated_0_01, updated_0_23],
        )
        inline_ptx(
            "st.global.v2.b32 [$0], {$1, $2};",
            read_only_args=[state_address_1, updated_1_01, updated_1_23],
        )

        acc_0 = state_0_1 * weight_0
        acc_0 = acc_0 + state_0_2 * weight_1
        acc_0 = acc_0 + state_0_3 * weight_2
        acc_0 = acc_0 + x_f32_0 * weight_3
        output[row_0, channel] = (acc_0 * sigmoid(acc_0)).to(cutlass.BFloat16)

        acc_1 = state_1_1 * weight_0
        acc_1 = acc_1 + state_1_2 * weight_1
        acc_1 = acc_1 + state_1_3 * weight_2
        acc_1 = acc_1 + x_f32_1 * weight_3
        output[row_1, channel] = (acc_1 * sigmoid(acc_1)).to(cutlass.BFloat16)


class CausalConv1dUpdateRowBatchKernel:
    """Host launcher for the guarded two-row, no-index specialization."""

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        state: cute.Tensor,
        output: cute.Tensor,
        state_indices: Optional[cute.Tensor],
        n_slots: cutlass.Int32,
        n_channels: cutlass.Int32,
        stream: cuda.CUstream,
    ) -> None:
        # ``state_indices`` and ``n_slots`` remain in the common compiled ABI.
        # API-side selection guarantees an unindexed N=128 descriptor with
        # enough identity-mapped slots before this specialization is compiled.
        _causal_conv1d_update_row_batch_kernel(
            x,
            weight,
            state,
            output,
            n_channels,
        ).launch(
            grid=(x.shape[0] // ROW_BATCH_ROWS, cute.ceil_div(x.shape[1], THREADS), 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )
