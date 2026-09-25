# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small CuTe utilities needed by the dedicated HSTU D=256 kernels."""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Boolean, Int32

SM100_TMEM_CAPACITY_COLUMNS = 512


def make_sm100_thread_cooperative_group(size: int):
    return cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, size)


class Sm100FmhaStaticTileSchedulerParams:
    def __init__(
        self,
        problem_shape_mbh: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        self.problem_shape_mbh = problem_shape_mbh
        self._loc = loc

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.problem_shape_mbh)
        self._values_count = len(values)
        return values

    def __new_from_mlir_values__(self, values):
        problem_shape = cutlass.new_from_mlir_values(self.problem_shape_mbh, values[: self._values_count])
        return Sm100FmhaStaticTileSchedulerParams(problem_shape, loc=self._loc)


class Sm100FmhaStaticTileScheduler:
    def __init__(
        self,
        params: Sm100FmhaStaticTileSchedulerParams,
        current_work_linear_idx: Int32,
        blk_coord: cute.Coord,
        grid_shape: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        self._params = params
        self._blk_coord = blk_coord
        self._grid_shape = grid_shape
        self._current_work_linear_idx = current_work_linear_idx
        self._is_first_block = True
        self._loc = loc

    @staticmethod
    def get_grid_shape(
        params: Sm100FmhaStaticTileSchedulerParams,
        *,
        loc=None,
        ip=None,
    ) -> cute.Shape:
        return params.problem_shape_mbh

    @staticmethod
    def check_valid_work_for_seqlen_q(
        q_tiler: int,
        current_idx: Int32,
        seqlen_q: Int32,
    ) -> Boolean:
        return current_idx * q_tiler < seqlen_q

    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        cur_tile_coord = (
            self._blk_coord[0],
            0,
            (self._blk_coord[1], self._blk_coord[2]),
        )
        return cutlass.utils.WorkTileInfo(cur_tile_coord, self._is_first_block)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._is_first_block = False
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def producer_tail(self, *, loc=None, ip=None):
        pass

    def __extract_mlir_values__(self):
        objects = (
            self._params,
            self._current_work_linear_idx,
            self._blk_coord,
            self._grid_shape,
        )
        values, self._values_pos = [], []
        for obj in objects:
            obj_values = cutlass.extract_mlir_values(obj)
            values.extend(obj_values)
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        objects = (
            self._params,
            self._current_work_linear_idx,
            self._blk_coord,
            self._grid_shape,
        )
        rebuilt = []
        for obj, count in zip(objects, self._values_pos):
            rebuilt.append(cutlass.new_from_mlir_values(obj, values[:count]))
            values = values[count:]
        return Sm100FmhaStaticTileScheduler(*rebuilt, loc=self._loc)


def compute_sm100_fmha_grid(
    output_shape: cute.Shape,
    cta_tiler: Tuple[int, int, int],
):
    params = Sm100FmhaStaticTileSchedulerParams(
        (
            cute.ceil_div(cute.size(output_shape[0]), cta_tiler[0]),
            cute.size(output_shape[2][0]),
            cute.size(output_shape[2][1]),
        ),
    )
    return params, Sm100FmhaStaticTileScheduler.get_grid_shape(params)


class HSTUFusedMask:
    """Mask-coordinate helpers for native HSTU D=256 backward."""

    @staticmethod
    @cute.jit
    def get_trip_start_count_via_block_info(
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        is_causal: cutlass.Constexpr[bool] = False,
        is_local: cutlass.Constexpr[bool] = False,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Tuple[Int32, Int32]:
        offset = seqlen_k - seqlen_q
        row_begin = blk_coord[0] * tile_shape[0]
        row_end = min((blk_coord[0] + 1) * tile_shape[0], seqlen_q)
        n_block_min = Int32(0)
        if cutlass.const_expr(is_local and window_size_left is not None):
            first_col = max(Int32(0), row_begin + offset - window_size_left)
            n_block_min = first_col // tile_shape[1]

        n_block_max = cute.ceil_div(seqlen_k, tile_shape[1])
        if cutlass.const_expr(is_causal or is_local):
            right = Int32(0) if window_size_right is None else window_size_right
            last_col_exclusive = min(seqlen_k, row_end + offset + right)
            n_block_max = cute.ceil_div(last_col_exclusive, tile_shape[1])
        n_block_max = max(n_block_min, n_block_max)
        return n_block_min, n_block_max - n_block_min

    @staticmethod
    @cute.jit
    def get_trip_mask_bounds_via_block_info(
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        is_causal: cutlass.Constexpr[bool] = False,
        is_local: cutlass.Constexpr[bool] = False,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Tuple[Int32, Int32]:
        start, _ = HSTUFusedMask.get_trip_start_count_via_block_info(
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            is_causal,
            is_local,
            window_size_left,
            window_size_right,
        )
        # Mask every visited tile. This is conservative and keeps the HSTU
        # predicates independent of softmax-specific fast bounds.
        return start, start

    @staticmethod
    @cute.jit
    def apply_mask_via_causal_local(
        predicates: cute.Tensor,
        index_qk: cute.Tensor,
        seqlen_q: Int32,
        seqlen_k: Int32,
        apply_semantic_window: cutlass.Constexpr[bool] = True,
        is_causal: cutlass.Constexpr[bool] = False,
        is_local: cutlass.Constexpr[bool] = False,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
        is_arbitrary: cutlass.Constexpr[bool] = False,
        func_num: cutlass.Constexpr[int] = 0,
        func: Optional[cute.Tensor] = None,
        query_offset: Int32 = Int32(0),
    ) -> None:
        offset = seqlen_k - seqlen_q
        for i in cutlass.range(cute.size(predicates), unroll_full=True):
            index_q, index_k = index_qk[i]
            valid = index_q < seqlen_q and index_k < seqlen_k
            if cutlass.const_expr(is_arbitrary):
                assert isinstance(func, cute.Tensor)
                func_row = query_offset + index_q
                value_valid = index_k < func[0, 0, func_row]
                for interval_idx in cutlass.range(func_num // 2, unroll_full=True):
                    interval_start = func[0, 2 * interval_idx + 1, func_row]
                    interval_end = func[0, 2 * interval_idx + 2, func_row]
                    value_valid = value_valid | ((index_k >= interval_start) & (index_k < interval_end))
                valid = valid and value_valid
            elif cutlass.const_expr(apply_semantic_window and (is_causal or is_local)):
                score_row = index_q + offset
                right = Int32(0) if window_size_right is None else window_size_right
                valid = valid and index_k <= score_row + right
                if cutlass.const_expr(is_local and window_size_left is not None):
                    valid = valid and index_k >= score_row - window_size_left
            predicates[i] = valid
