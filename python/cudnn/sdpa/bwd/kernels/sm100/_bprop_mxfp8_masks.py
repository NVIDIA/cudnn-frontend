# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Fused mask helpers for the SM100 D256 MXFP8 SDPA-backward kernels.

``fmha_cute/masks.py`` of the source repo (itself the CUTLASS CuTe DSL FMHA
example's mask module extended with the *_BWD variants).

Ported from Xinbo Zhao's fmha_mxfp8_large_head_dim (2026-09-01).
Kept as close to the source as the package rules allow so upstream fixes
stay diff-able; only the imports and this note differ.
"""

import enum
from typing import Tuple, Optional
import cutlass

from cutlass.cutlass_dsl import (
    Int32,
    Float32,
    min,
)
import cutlass.cute as cute

##############################################################################
# Fused Mask
##############################################################################


class MaskEnum(enum.Enum):
    """Enumeration of mask types for FMHA operations.

    - RESIDUAL_MASK: Residual mask for handling variable sequence lengths
    - WINDOW_MASK: Window mask for attention which also includes causal and no mask
    - WINDOW_MASK_INFERENCE: Same as the window mask, but has the limitation that the end of q is aligned with the end of k
    - WINDOW_MASK_BWD: Window mask for backward pass
    - WINDOW_MASK_BWD_INFERENCE: Same as the window mask for backward pass, but has the limitation that the end of q is aligned with the end of k
    """

    RESIDUAL_MASK = enum.auto()
    RESIDUAL_MASK_BWD = enum.auto()
    WINDOW_MASK = enum.auto()
    WINDOW_MASK_INFERENCE = enum.auto()
    WINDOW_MASK_BWD = enum.auto()
    WINDOW_MASK_BWD_INFERENCE = enum.auto()


class FusedMask:
    """A fused mask implementation for FMHA operations.

    This class handles different types of attention masks including no mask,
    residual mask for variable sequence lengths, and causal mask for
    autoregressive attention patterns.

    The class provides methods to:
    - Calculate trip counts for different mask types
    - Apply masks to attention scores
    - Handle masked and unmasked trip calculations
    """

    def get_trip_count(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Int32:
        """
        Calculate the number of trips needed for the current block.

        The trip count depends on the mask type and the block coordinates.
        For causal masks, it considers the autoregressive constraint.

        :param mask_type: Type of mask to use
        :type mask_type: utils.MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]

        :return: Number of trips needed.
        :rtype: Int32
        """
        result = 0
        offset = 0
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        if cutlass.const_expr(mask_type == MaskEnum.RESIDUAL_MASK):
            result = cute.ceil_div(seqlen_k, tile_shape[1])
        if cutlass.const_expr(mask_type is MaskEnum.RESIDUAL_MASK_BWD):
            result = cute.ceil_div(seqlen_q, tile_shape[0])
        if cutlass.const_expr(mask_type == MaskEnum.WINDOW_MASK or mask_type == MaskEnum.WINDOW_MASK_INFERENCE):
            if cutlass.const_expr(window_size_right is None):
                result = cute.ceil_div(seqlen_k, tile_shape[1])
            else:
                max_idx_q = (blk_coord[0] + 1) * tile_shape[0]
                idx_k = max_idx_q + offset + window_size_right
                tmp_blocks_k = cute.ceil_div(idx_k, tile_shape[1])
                max_blocks_k = cute.ceil_div(seqlen_k, tile_shape[1])
                result = min(max_blocks_k, tmp_blocks_k)
        if cutlass.const_expr(mask_type == MaskEnum.WINDOW_MASK_BWD or mask_type == MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            if cutlass.const_expr(window_size_left is None):
                result = cute.ceil_div(seqlen_q, tile_shape[0])
            else:
                max_idx_k = (blk_coord[1] + 1) * tile_shape[1]
                idx_k = max_idx_k + offset + window_size_left
                tmp_blocks_q = cute.ceil_div(idx_k, tile_shape[0])
                max_blocks_q = cute.ceil_div(seqlen_q, tile_shape[0])
                result = min(max_blocks_q, tmp_blocks_q)
        start_block = FusedMask.get_trip_start(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        result = result - start_block
        return result

    @cute.jit
    def get_trip_start(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Int32:
        """
        Get the start of the trip for the current block.

        :param mask_type: Type of mask to use
        :type mask_type: utils.MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]
        """
        result = 0
        offset = 0
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK or mask_type is MaskEnum.WINDOW_MASK_INFERENCE):
            if cutlass.const_expr(window_size_left is not None):
                min_idx_q = blk_coord[0] * tile_shape[0]
                idx_k = min_idx_q + offset - window_size_left
                tmp_blocks_k = idx_k // tile_shape[1]
                result = max(tmp_blocks_k, result)
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_BWD or mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            if cutlass.const_expr(window_size_right is not None):
                min_idx_k = blk_coord[1] * tile_shape[1]
                idx_q = min_idx_k + offset - window_size_right
                tmp_blocks_q = idx_q // tile_shape[0]
                result = max(tmp_blocks_q, result)
        return result

    @cute.jit
    def get_leading_mask_id(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Tuple[Int32, Int32]:
        """
        Get the begin and end tile idx for the leading mask.

        :param mask_type: Type of mask to use
        :type mask_type: utils.MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]

        :return: Tuple of (begin, end) tile idx for the leading mask.
        :rtype: Tuple[Int32, Int32]
        """
        offset = 0
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        leading_mask_begin = FusedMask.get_trip_start(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        trip_count = FusedMask.get_trip_count(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )

        leading_mask_end = leading_mask_begin
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK or mask_type is MaskEnum.WINDOW_MASK_INFERENCE):
            if cutlass.const_expr(window_size_left is not None):
                min_idx_q = (blk_coord[0] + 1) * tile_shape[0] + offset - window_size_left
                leading_mask_end = min(
                    cute.ceil_div(min_idx_q, tile_shape[1]) - 1,
                    trip_count + leading_mask_begin - 1,
                )
            else:
                leading_mask_end = leading_mask_begin - 1
        elif cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_BWD or mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            if cutlass.const_expr(window_size_right is not None):
                min_idx_k = (blk_coord[1] + 1) * tile_shape[1] + offset - window_size_right
                leading_mask_end = cute.ceil_div(min_idx_k, tile_shape[0]) - 1
            else:
                leading_mask_end = leading_mask_begin - 1
        return leading_mask_begin, leading_mask_end

    @cute.jit
    def get_trailing_mask_id(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Tuple[Optional[Int32], Optional[Int32]]:
        """
        Get the begin and end tile idx for the trailing mask.

        :param mask_type: Type of mask to use
        :type mask_type: utils.MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]

        :return: Tuple of (begin, end) tile idx for the trailing mask.
        :rtype: Tuple[Int32, Int32]
        """
        offset = 0
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        trip_start = FusedMask.get_trip_start(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        trip_count = FusedMask.get_trip_count(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )

        trailing_mask_begin, trailing_mask_end = None, None
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK or mask_type is MaskEnum.WINDOW_MASK_INFERENCE):
            if cutlass.const_expr(window_size_right is not None):
                min_idx_q = blk_coord[0] * tile_shape[0] + offset + window_size_right
                trailing_mask_begin = min(min_idx_q // tile_shape[1], trip_count + trip_start - 1)
                trailing_mask_end = trip_count + trip_start - 1
            else:
                # last tile, we always apply mask on it regardless whether it's a residual tile
                trailing_mask_begin = trip_count + trip_start - 1
                trailing_mask_end = trip_count + trip_start - 1
        else:
            if cutlass.const_expr(window_size_left is not None):
                min_idx_k = blk_coord[1] * tile_shape[1] + offset + window_size_left + 1
                max_idx_k = (blk_coord[1] + 1) * tile_shape[1] + offset + window_size_left
                trailing_mask_begin = min(
                    cute.ceil_div(min_idx_k, tile_shape[0]) - 1,
                    trip_count + trip_start - 1,
                )
                trailing_mask_end = min(
                    cute.ceil_div(max_idx_k, tile_shape[0]) - 1,
                    trip_count + trip_start - 1,
                )
            else:
                # last tile, we always apply mask on it regardless whether it's a residual tile
                trailing_mask_begin = trip_count + trip_start - 1
                trailing_mask_end = trip_count + trip_start - 1

        return trailing_mask_begin, trailing_mask_end

    @cute.jit
    def get_masked_leading_count(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Int32:
        """
        Calculate the number of masked trips for the leading mask.

        This is used for blocks that need special handling due to masking.

        :param mask_type: Type of mask to use
        :type mask_type: utils.MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]

        :return: Number of masked trips.
        :rtype: Int32
        """
        result = 0
        if cutlass.const_expr(mask_type is not MaskEnum.RESIDUAL_MASK and mask_type is not MaskEnum.RESIDUAL_MASK_BWD):
            if cutlass.const_expr(window_size_left is not None or window_size_right is not None):
                leading_mask_begin, leading_mask_end = FusedMask.get_leading_mask_id(
                    mask_type,
                    blk_coord,
                    tile_shape,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                result = max(leading_mask_end - leading_mask_begin + 1, 0)

        return result

    @cute.jit
    def get_masked_trailing_count(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
        rem_count: Optional[Int32] = 0,
    ) -> Int32:
        """
        Calculate the number of masked trips for the trailing mask.

        This is used for blocks that need special handling due to masking.

        :param mask_type: Type of mask to use
        :type mask_type: utils.MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]
        :param rem_count: Remaining count from previous calculations.
        :type rem_count: Int32

        :return: Number of masked trips.
        :rtype: Int32
        """
        result = 0

        if cutlass.const_expr(mask_type is not MaskEnum.RESIDUAL_MASK and mask_type is not MaskEnum.RESIDUAL_MASK_BWD):
            if cutlass.const_expr(window_size_left is not None or window_size_right is not None):
                trailing_mask_begin, trailing_mask_end = FusedMask.get_trailing_mask_id(
                    mask_type,
                    blk_coord,
                    tile_shape,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                leading_mask_begin, leading_mask_end = FusedMask.get_leading_mask_id(
                    mask_type,
                    blk_coord,
                    tile_shape,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                if cutlass.const_expr(trailing_mask_begin is not None and trailing_mask_end is not None):
                    if trailing_mask_begin <= leading_mask_end:
                        result = max(trailing_mask_end - leading_mask_end, 0)
                    else:
                        result = max(trailing_mask_end - trailing_mask_begin + 1, 0)
        else:
            if seqlen_k % tile_shape[1] != 0:
                result = 1
            else:
                result = 0

        return result + rem_count

    @cute.jit
    def get_unmasked_trip_count(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Int32:
        """
        Calculate the number of unmasked trips for the current block.

        This represents the number of trips that don't require special
        masking treatment.

        :param mask_type: Type of mask to use
        :type mask_type: utils.MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]

        :return: Number of unmasked trips.
        :rtype: Int32
        """
        result = (
            FusedMask.get_trip_count(
                mask_type,
                blk_coord,
                tile_shape,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
            )
            - FusedMask.get_masked_leading_count(
                mask_type,
                blk_coord,
                tile_shape,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
            )
            - FusedMask.get_masked_trailing_count(
                mask_type,
                blk_coord,
                tile_shape,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
                0,
            )
        )
        return result

    @cute.jit
    def apply_mask(
        mask_type: MaskEnum,
        acc_qk: cute.Tensor,
        index_qk: cute.Tensor,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[int] = None,
        window_size_right: Optional[int] = None,
        index_transform: cutlass.Constexpr = lambda index_q, index_k: (
            index_q,
            index_k,
        ),
    ):
        """
        Apply the appropriate mask to the attention scores.

        This method modifies the attention scores (acc_qk) based on the mask type
        and the positions in the index tensor.

        :param mask_type: Type of mask to use
        :type mask_type: utils.MaskEnum
        :param acc_qk: Accumulated QK attention scores tensor.
        :type acc_qk: cute.Tensor
        :param index_qk: Index tensor containing position information.
        :type index_qk: cute.Tensor
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Optional[int]
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[int]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[int]
        """

        tidx, tidy, tidx = cute.arch.thread_idx()
        offset = 0
        offset = (
            seqlen_k - seqlen_q if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_INFERENCE or mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE) else 0
        )
        for i in cutlass.range_constexpr(cute.size(acc_qk)):
            index_q, index_k = index_transform(*index_qk[i])
            if cutlass.const_expr(window_size_left is not None or window_size_right is not None):
                if cutlass.const_expr(window_size_left is None):
                    if index_q + offset + window_size_right < index_k:
                        acc_qk[i] = -Float32.inf
                    if index_k >= seqlen_k or index_q >= seqlen_q:  # residual mask
                        acc_qk[i] = -Float32.inf
                elif cutlass.const_expr(window_size_right is None):
                    if index_q + offset - window_size_left > index_k:
                        acc_qk[i] = -Float32.inf
                    if index_k >= seqlen_k or index_q >= seqlen_q:  # residual mask
                        acc_qk[i] = -Float32.inf
                else:
                    max_K_index = min(index_q + offset + window_size_right, seqlen_k)
                    min_K_index = max(0, index_q + offset - window_size_left)
                    if index_k > max_K_index or index_k < min_K_index:
                        acc_qk[i] = -Float32.inf
                    if index_k >= seqlen_k or index_q >= seqlen_q:  # residual mask
                        acc_qk[i] = -Float32.inf

            if cutlass.const_expr(mask_type == MaskEnum.RESIDUAL_MASK or mask_type == MaskEnum.RESIDUAL_MASK_BWD):
                if index_k >= seqlen_k or index_q >= seqlen_q:
                    acc_qk[i] = -Float32.inf
