# Copyright (c) 2025, Tri Dao.

from typing import Optional
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute

from cutlass.cute.typing import Int32, Tuple
from .utils import split_wg


@dataclass(frozen=True)
class AttentionMask:
    kBlockM: cutlass.Constexpr[int]
    kBlockN: cutlass.Constexpr[int]
    cta_tiler: cutlass.Constexpr[Tuple[int, int]]
    is_arbitrary: cutlass.Constexpr[bool]
    is_causal: cutlass.Constexpr[bool]
    is_local: cutlass.Constexpr[bool]
    is_context: cutlass.Constexpr[bool]
    is_target: cutlass.Constexpr[bool]
    target_group_size: cutlass.Constexpr[int]
    func_num: cutlass.Constexpr[int]
    window_size_left: cutlass.Constexpr[int]
    window_size_right: cutlass.Constexpr[int]
    offset_q: cutlass.Constexpr[int]
    seqlen_q: cutlass.Constexpr[int]
    seqlen_k: cutlass.Constexpr[int]
    seqlen_c: cutlass.Constexpr[int]
    seqlen_h: cutlass.Constexpr[int]
    offset_dynamic: cutlass.Constexpr[int]
    func: Optional[cute.Tensor] # (n_func, L_func)
    swapAB: cutlass.Constexpr[bool]

    # when seqlen_q = seqlen_k && tilesize is square, we only need mask first n_tile, preds can be calculated in compile time
    @cute.jit
    def apply_mask_causal(
        self,
        preds: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
    ) -> None:
        cS = cute.make_identity_tensor((self.kBlockM, self.kBlockN))
        tScS = thr_mma.partition_C(cS)
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        row_id, col_id = (1, 0) if cutlass.const_expr(self.swapAB) else (0, 1)

        for i in cutlass.range_constexpr(cute.size(preds), unroll_full=True):
            preds[i] = True

        for i in cutlass.range_constexpr(cute.size(preds), unroll_full=True):
            block_row = cute.get(tScS_t2r[i], mode=[row_id])
            block_col = cute.get(tScS_t2r[i], mode=[col_id])
            if block_col > block_row:
                preds[i] = False


    @cute.jit
    def apply_mask(
        self,
        preds: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
        mask_target: cutlass.Constexpr[bool] = False,
        mask_history: cutlass.Constexpr[bool] = False,
        mask_paged: cutlass.Constexpr[int] = 0,
    ) -> None:
        seqlen_offset = self.seqlen_k - self.seqlen_q
        cS = cute.make_identity_tensor((self.kBlockM, self.kBlockN))
        tScS = thr_mma.partition_C(cS)
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        base_row = m_block * self.kBlockM + seqlen_offset - self.offset_dynamic
        base_col = n_block * self.kBlockN
        row_id, col_id = (1, 0) if cutlass.const_expr(self.swapAB) else (0, 1)

        limit_right = lambda row: min(self.seqlen_k, row + 1 + self.window_size_right)
        limit_left = lambda row: max(0, row - self.window_size_left)

        block_row = cute.get(tScS_t2r[0], mode=[row_id])
        row = block_row + base_row

        target_index = (row - self.seqlen_h) // self.target_group_size if cutlass.const_expr(mask_target) else 0
        target_col_limit_left = self.seqlen_h + target_index * self.target_group_size if cutlass.const_expr(mask_target) else 0

        col_limit_right = limit_right(row)
        col_limit_left = limit_left(row)
        if cutlass.const_expr(mask_paged == 1):
            base_col += self.seqlen_h
        elif cutlass.const_expr(mask_paged == -1):
            col_limit_right = min(col_limit_right, self.seqlen_h)

        for i in cutlass.range_constexpr(cute.size(preds), unroll_full=True):
            preds[i] = True

        for i in cutlass.range_constexpr(cute.size(preds), unroll_full=True):
            block_col = cute.get(tScS_t2r[i], mode=[col_id])
            col = block_col + base_col

            if cutlass.const_expr(not self.is_causal and not self.is_local and not self.is_arbitrary):
                if col >= self.seqlen_k:
                    preds[i] = False
            elif cutlass.const_expr(mask_history):
                if col >= self.seqlen_h:
                    preds[i] = False
            elif cutlass.const_expr(self.is_arbitrary):
                func_row = row + self.offset_q - seqlen_offset
                for j in cutlass.range(self.func_num // 2, unroll_full=True):
                    if col >= self.func[2 * j, func_row].value and col < self.func[2 * j + 1, func_row].value:
                        preds[i] = False
                if col >= self.func[self.func_num - 1, func_row].value or col >= self.seqlen_k:
                    preds[i] = False
            else:
                if col >= col_limit_right: # causal
                    preds[i] = False
                if cutlass.const_expr(self.is_local):
                    if col < col_limit_left:
                        preds[i] = False
                if cutlass.const_expr(mask_target):
                    if row >= self.seqlen_h and col >= self.seqlen_h and col < target_col_limit_left:
                    # i think we could remove row >= self.seqlen_h condition, but get worse performance (102us vs 96us)
                    # if col >= self.seqlen_h and col < target_col_limit_left:
                        preds[i] = False
                if cutlass.const_expr(self.is_context):
                    if row < self.seqlen_c and col < self.seqlen_h:
                        preds[i] = True

    @cute.jit
    def apply_mask_swapAB(
        self,
        preds: cute.Tensor,
        wg_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
        mask_casual: cutlass.Constexpr[bool] = False,
        mask_target: cutlass.Constexpr[bool] = False,
        mask_seqlen: cutlass.Constexpr[bool] = False,
    ) -> None:
        seqlen_offset = self.seqlen_k - self.seqlen_q
        cS = cute.make_identity_tensor((self.kBlockM, self.kBlockN))
        tScS = thr_mma.partition_C(cS)
        tScS_t2r = thr_tmem_load.partition_D(cS)
        tScS_t2r = split_wg(tScS_t2r, 2, wg_idx)
        base_row = m_block * self.kBlockM + seqlen_offset
        base_col = n_block * self.kBlockN
        row_id, col_id = (1, 0) if cutlass.const_expr(self.swapAB) else (0, 1)

        col_limit_right = lambda row: min(self.seqlen_k, row + 1 + self.window_size_right)
        col_limit_left = lambda row: max(0, row - self.window_size_left)

        for i in cutlass.range(cute.size(preds), unroll_full=True):
            preds[i] = True

        for i in cutlass.range(cute.size(preds), unroll_full=True):
            block_row = cute.get(tScS_t2r[i], mode=[row_id])
            row = block_row + base_row
            # target_index = (row - self.seqlen_h) // self.target_group_size if cutlass.const_expr(self.is_target) else 0
            # target_col_limit_left = self.seqlen_h + target_index * self.target_group_size if cutlass.const_expr(self.is_target) else 0
            target_index = (row - self.seqlen_h) // self.target_group_size if cutlass.const_expr(mask_target) else 0
            target_col_limit_left = self.seqlen_h + target_index * self.target_group_size if cutlass.const_expr(mask_target) else 0

            block_col = cute.get(tScS_t2r[i], mode=[col_id])
            col = block_col + base_col

            if col >= self.seqlen_k or row >= self.seqlen_q + seqlen_offset and mask_seqlen:
                preds[i] = False
            if cutlass.const_expr(self.is_arbitrary):
                func_row = row + self.offset_q - seqlen_offset
                for j in cutlass.range(self.func_num // 2, unroll_full=True):
                    if col >= self.func[2*j, func_row].value and col < self.func[2*j+1, func_row].value:
                        preds[i] = False
                if col >= self.func[self.func_num - 1, func_row].value:
                    preds[i] = False
            else:
                if col >= col_limit_right(row) and mask_casual:
                    preds[i] = False
                if cutlass.const_expr(self.is_local):
                    if col < col_limit_left(row):
                        preds[i] = False
                # if cutlass.const_expr(self.is_target):
                if cutlass.const_expr(mask_target):
                    if row >= self.seqlen_h and col >= self.seqlen_h and col < target_col_limit_left:
                        preds[i] = False
                if cutlass.const_expr(self.is_context):
                    if row < self.seqlen_c and col < self.seqlen_h:
                        preds[i] = True
