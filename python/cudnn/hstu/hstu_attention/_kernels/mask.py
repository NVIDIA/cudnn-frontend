# Copyright (c) 2025, Tri Dao.
# SPDX-License-Identifier: MIT

from typing import Optional
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute

from cutlass.cute.typing import Boolean, Int32
from .utils import shl_b32, shr_u32, split_wg

_R2P_CHUNK_SIZE = 32
_BWD_ENDPOINT_PREFETCH_CHUNK = 8


@cute.jit
def _r2p_bitmask_below(limit: Int32, chunk: int) -> cutlass.Uint32:
    """Return a keep mask for local columns strictly below ``limit``."""

    shift = max((chunk + 1) * _R2P_CHUNK_SIZE - limit, 0)
    return shr_u32(cutlass.Uint32(0xFFFFFFFF), cutlass.Uint32(shift))


@cute.jit
def _r2p_bitmask_above(limit: Int32, chunk: int) -> cutlass.Uint32:
    """Return a keep mask for local columns at or above ``limit``."""

    shift = max(limit - chunk * _R2P_CHUNK_SIZE, 0)
    return shl_b32(cutlass.Uint32(0xFFFFFFFF), cutlass.Uint32(shift))


@dataclass(frozen=True)
class AttentionMask:
    kBlockM: cutlass.Constexpr[int]
    kBlockN: cutlass.Constexpr[int]
    is_arbitrary: cutlass.Constexpr[bool]
    is_causal: cutlass.Constexpr[bool]
    is_local: cutlass.Constexpr[bool]
    func_num: cutlass.Constexpr[int]
    window_size_left: cutlass.Constexpr[int]
    window_size_right: cutlass.Constexpr[int]
    offset_q: cutlass.Constexpr[int]
    seqlen_q: cutlass.Constexpr[int]
    seqlen_k: cutlass.Constexpr[int]
    offset_dynamic: cutlass.Constexpr[int]
    func: Optional[cute.Tensor]  # (n_func, L_func)
    swapAB: cutlass.Constexpr[bool]

    @cute.jit
    def apply_mask(
        self,
        preds: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
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
        q_row = row - seqlen_offset
        q_in_bounds = Boolean(q_row >= 0 and q_row < self.seqlen_q)

        col_limit_right = limit_right(row)
        col_limit_left = limit_left(row)

        for i in cutlass.range_constexpr(cute.size(preds), unroll_full=True):
            preds[i] = True

        for i in cutlass.range_constexpr(cute.size(preds), unroll_full=True):
            block_col = cute.get(tScS_t2r[i], mode=[col_id])
            col = block_col + base_col

            if cutlass.const_expr(not self.is_causal and not self.is_local and not self.is_arbitrary):
                if col >= self.seqlen_k:
                    preds[i] = False
            elif cutlass.const_expr(self.is_arbitrary):
                if q_in_bounds:
                    func_row = q_row + self.offset_q
                    # Keep the interval union:
                    # [0, F0) U [F1, F2) U [F3, F4) ...
                    preds[i] = Boolean(col < self.seqlen_k and col < self.func[0, func_row].value)
                    for interval in cutlass.range(
                        self.func_num // 2,
                        unroll_full=True,
                    ):
                        if col < self.seqlen_k and col >= self.func[2 * interval + 1, func_row].value and col < self.func[2 * interval + 2, func_row].value:
                            preds[i] = True
                else:
                    preds[i] = False
            else:
                if col >= col_limit_right:  # causal
                    preds[i] = False
                if cutlass.const_expr(self.is_local):
                    if col < col_limit_left:
                        preds[i] = False

    @cute.jit
    def build_mask_r2p(
        self,
        keep_masks: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
    ) -> None:
        """Build four 32-bit keep masks for one SM100 forward score row.

        With the forward ``Ld32x32bOp(Repetition(32))`` mapping, every thread
        owns one Q row and its flattened score fragment walks the 128 K
        columns in order.  Keeping the masks in four GPRs avoids carrying 128
        Boolean predicates across the TMEM wait and SiLU computation.
        """

        cS = cute.make_identity_tensor((self.kBlockM, self.kBlockN))
        tScS = thr_mma.partition_C(cS)
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        num_scores = cutlass.const_expr(cute.size(tScS_t2r))
        num_chunks = cutlass.const_expr(cute.size(keep_masks))
        assert num_scores == num_chunks * _R2P_CHUNK_SIZE

        base_row = m_block * self.kBlockM - self.offset_dynamic
        base_col = n_block * self.kBlockN
        block_row = cute.get(tScS_t2r[0], mode=[0])
        q_row = block_row + base_row
        q_in_bounds = Boolean(q_row >= 0 and q_row < self.seqlen_q)

        for chunk in cutlass.range_constexpr(num_chunks, unroll_full=True):
            keep_masks[chunk] = cutlass.Uint32(0)

        # A packed residual Q tile can overlap the next sequence.  Guard the
        # func loads themselves, not just the resulting score predicates.
        if q_in_bounds:
            if cutlass.const_expr(self.is_arbitrary):
                func_row = q_row + self.offset_q
                col_limits = cute.make_rmem_tensor(
                    (self.func_num,),
                    cutlass.Int32,
                )
                for endpoint in cutlass.range_constexpr(
                    self.func_num,
                    unroll_full=True,
                ):
                    col_limits[endpoint] = max(
                        self.func[endpoint, func_row].value - base_col,
                        0,
                    )
                seqlen_k_limit = max(self.seqlen_k - base_col, 0)

                for chunk in cutlass.range_constexpr(
                    num_chunks,
                    unroll_full=True,
                ):
                    # Construct the interval union directly with R2P masks:
                    # [0, F0) U [F1, F2) U [F3, F4) ...
                    combined_mask = _r2p_bitmask_below(
                        col_limits[0],
                        chunk,
                    )
                    for interval in cutlass.range_constexpr(
                        self.func_num // 2,
                        unroll_full=True,
                    ):
                        interval_mask = _r2p_bitmask_above(
                            col_limits[2 * interval + 1],
                            chunk,
                        ) & _r2p_bitmask_below(
                            col_limits[2 * interval + 2],
                            chunk,
                        )
                        combined_mask = combined_mask | interval_mask
                    # HSTU packed-varlen tiles still need the physical K bound,
                    # which is independent of the arbitrary interval metadata.
                    keep_masks[chunk] = combined_mask & _r2p_bitmask_below(
                        seqlen_k_limit,
                        chunk,
                    )
            else:
                seqlen_offset = self.seqlen_k - self.seqlen_q
                col_limit = min(
                    self.seqlen_k,
                    q_row + seqlen_offset + 1 + self.window_size_right,
                )
                col_limit = max(col_limit - base_col, 0)
                for chunk in cutlass.range_constexpr(
                    num_chunks,
                    unroll_full=True,
                ):
                    keep_masks[chunk] = _r2p_bitmask_below(
                        col_limit,
                        chunk,
                    )

    @cute.jit
    def apply_mask_seqlen(
        self,
        preds: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
    ) -> None:
        """Apply only packed-varlen Q/K bounds, without reading ``func``."""

        cS = cute.make_identity_tensor((self.kBlockM, self.kBlockN))
        tScS = thr_mma.partition_C(cS)
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        row_id, col_id = (1, 0) if cutlass.const_expr(self.swapAB) else (0, 1)
        base_row = m_block * self.kBlockM - self.offset_dynamic
        base_col = n_block * self.kBlockN

        for i in cutlass.range_constexpr(cute.size(preds), unroll_full=True):
            block_row = cute.get(tScS_t2r[i], mode=[row_id])
            block_col = cute.get(tScS_t2r[i], mode=[col_id])
            preds[i] = Boolean(block_row + base_row >= 0 and block_row + base_row < self.seqlen_q and block_col + base_col < self.seqlen_k)

    @cute.jit
    def apply_mask_swapAB(
        self,
        preds: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        tScS_t2r: cute.Tensor,
        mask_causal: cutlass.Constexpr[bool] = False,
        mask_seqlen: cutlass.Constexpr[bool] = False,
    ) -> None:
        seqlen_offset = self.seqlen_k - self.seqlen_q
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
            q_row = row - seqlen_offset

            block_col = cute.get(tScS_t2r[i], mode=[col_id])
            col = block_col + base_col

            if col >= self.seqlen_k or row >= self.seqlen_q + seqlen_offset and mask_seqlen:
                preds[i] = False
            if cutlass.const_expr(self.is_arbitrary):
                q_in_bounds = Boolean(q_row >= 0 and q_row < self.seqlen_q)
                # A residual Q tile may contain rows from the next packed
                # sequence.  Do not even form a func load for those rows.
                if q_in_bounds:
                    func_row = q_row + self.offset_q
                    # Keep the interval union:
                    # [0, F0) U [F1, F2) U [F3, F4) ...
                    preds[i] = Boolean(col < self.seqlen_k and col < self.func[0, func_row].value)
                    for interval in cutlass.range(
                        self.func_num // 2,
                        unroll_full=True,
                    ):
                        if col < self.seqlen_k and col >= self.func[2 * interval + 1, func_row].value and col < self.func[2 * interval + 2, func_row].value:
                            preds[i] = True
                else:
                    preds[i] = False
            else:
                if col >= col_limit_right(row) and mask_causal:
                    preds[i] = False
                if cutlass.const_expr(self.is_local):
                    if col < col_limit_left(row):
                        preds[i] = False

    @cute.jit
    def build_mask_swapAB_r2p(
        self,
        keep_masks: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        tScS_t2r: cute.Tensor,
    ) -> None:
        """Build packed predicates for the transposed causal backward tile."""

        assert self.is_causal and not self.is_local and not self.is_arbitrary
        row_id, col_id = (1, 0) if cutlass.const_expr(self.swapAB) else (0, 1)
        assert cute.size(tScS_t2r) == cute.size(keep_masks) * _R2P_CHUNK_SIZE

        thread_row_offset = cute.get(tScS_t2r[0], mode=[row_id])
        thread_col_offset = cute.get(tScS_t2r[0], mode=[col_id])
        seqlen_q_row_limit = self.seqlen_q - m_block * self.kBlockM - thread_row_offset
        seqlen_k_col_limit = self.seqlen_k - n_block * self.kBlockN - thread_col_offset
        row_limit_lower = seqlen_q_row_limit - seqlen_k_col_limit
        row_limit_upper = seqlen_q_row_limit

        num_rep = cutlass.const_expr(cute.size(tScS_t2r, mode=[0]))
        num_warp_groups = 2
        num_scores = cutlass.const_expr(cute.size(tScS_t2r))
        row_limit_lower = row_limit_lower // (num_rep * num_warp_groups) * num_rep + min(row_limit_lower % (num_rep * num_warp_groups), num_rep)
        row_limit_upper = row_limit_upper // (num_rep * num_warp_groups) * num_rep + min(row_limit_upper % (num_rep * num_warp_groups), num_rep)
        row_limit_lower = min(max(row_limit_lower, 0), num_scores)
        row_limit_upper = min(max(row_limit_upper, 0), num_scores)
        for chunk in cutlass.range_constexpr(cute.size(keep_masks)):
            keep_mask = _r2p_bitmask_above(
                row_limit_lower,
                chunk,
            ) & _r2p_bitmask_below(
                row_limit_upper,
                chunk,
            )
            keep_masks[chunk] = keep_mask if seqlen_k_col_limit > 0 else cutlass.Uint32(0)

    @cute.jit
    def apply_mask_swapAB_arbitrary_prefetch(
        self,
        preds: cute.Tensor,
        wg_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_tmem_load: cute.TiledCopy,
    ) -> None:
        """Apply a transposed arbitrary mask with 8-way endpoint-load MLP.

        Fused backward maps every score element to a different Q row, so
        endpoints cannot be shared as a forward-style R2P mask.  Endpoint
        planes are loaded eight rows at a time before they are consumed.  Two
        eight-element caches bound register pressure independently of
        ``func_num`` while preserving the interval-union semantics:
        ``[0, F0) U [F1, F2) U [F3, F4) ...``.
        """

        cS = cute.make_identity_tensor((self.kBlockM, self.kBlockN))
        tScS_t2r = thr_tmem_load.partition_D(cS)
        tScS_t2r = split_wg(tScS_t2r, 2, wg_idx)
        row_id, col_id = (1, 0) if cutlass.const_expr(self.swapAB) else (0, 1)
        base_row = m_block * self.kBlockM
        base_col = n_block * self.kBlockN
        num_scores = cutlass.const_expr(cute.size(preds))
        endpoint_begin = cute.make_rmem_tensor(
            (_BWD_ENDPOINT_PREFETCH_CHUNK,),
            cutlass.Int32,
        )
        endpoint_end = cute.make_rmem_tensor(
            (_BWD_ENDPOINT_PREFETCH_CHUNK,),
            cutlass.Int32,
        )
        for local_idx in cutlass.range_constexpr(
            _BWD_ENDPOINT_PREFETCH_CHUNK,
            unroll_full=True,
        ):
            endpoint_begin[local_idx] = Int32(0)
            endpoint_end[local_idx] = Int32(0)

        for base in cutlass.range_constexpr(
            0,
            num_scores,
            _BWD_ENDPOINT_PREFETCH_CHUNK,
        ):
            chunk = cutlass.const_expr(min(_BWD_ENDPOINT_PREFETCH_CHUNK, num_scores - base))

            # Prefetch F0 for eight independent Q rows.
            for local_idx in cutlass.range_constexpr(chunk):
                score_idx = base + local_idx
                block_row = cute.get(tScS_t2r[score_idx], mode=[row_id])
                block_col = cute.get(tScS_t2r[score_idx], mode=[col_id])
                q_row = block_row + base_row
                k_col = block_col + base_col
                score_in_bounds = Boolean(q_row >= 0 and q_row < self.seqlen_q and k_col < self.seqlen_k)
                if score_in_bounds:
                    endpoint_end[local_idx] = self.func[0, q_row + self.offset_q].value

            for local_idx in cutlass.range_constexpr(chunk):
                score_idx = base + local_idx
                block_row = cute.get(tScS_t2r[score_idx], mode=[row_id])
                block_col = cute.get(tScS_t2r[score_idx], mode=[col_id])
                q_row = block_row + base_row
                k_col = block_col + base_col
                valid = Boolean(q_row >= 0 and q_row < self.seqlen_q and k_col < self.seqlen_k)
                if valid:
                    valid = Boolean(k_col < endpoint_end[local_idx])
                preds[score_idx] = valid

            # Prefetch and add one kept interval at a time.  This keeps at most
            # 16 endpoint values live even for func_num=7.
            for interval in cutlass.range_constexpr(self.func_num // 2, unroll_full=True):
                for local_idx in cutlass.range_constexpr(chunk):
                    score_idx = base + local_idx
                    block_row = cute.get(tScS_t2r[score_idx], mode=[row_id])
                    block_col = cute.get(tScS_t2r[score_idx], mode=[col_id])
                    q_row = block_row + base_row
                    k_col = block_col + base_col
                    score_in_bounds = Boolean(q_row >= 0 and q_row < self.seqlen_q and k_col < self.seqlen_k)
                    if score_in_bounds:
                        func_row = q_row + self.offset_q
                        endpoint_begin[local_idx] = self.func[2 * interval + 1, func_row].value
                        endpoint_end[local_idx] = self.func[2 * interval + 2, func_row].value

                for local_idx in cutlass.range_constexpr(chunk):
                    score_idx = base + local_idx
                    block_row = cute.get(tScS_t2r[score_idx], mode=[row_id])
                    block_col = cute.get(tScS_t2r[score_idx], mode=[col_id])
                    q_row = block_row + base_row
                    k_col = block_col + base_col
                    score_in_bounds = Boolean(q_row >= 0 and q_row < self.seqlen_q and k_col < self.seqlen_k)
                    if score_in_bounds:
                        if k_col >= endpoint_begin[local_idx] and k_col < endpoint_end[local_idx]:
                            preds[score_idx] = True

    @cute.jit
    def apply_mask_swapAB_seqlen(
        self,
        preds: cute.Tensor,
        wg_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_tmem_load: cute.TiledCopy,
    ) -> None:
        """Apply only packed Q/K bounds to the backward split-WG fragment."""

        cS = cute.make_identity_tensor((self.kBlockM, self.kBlockN))
        # Backward's TMEM copy partitions the identity tile directly before
        # splitting it across the two compute warpgroups.
        tScS_t2r = thr_tmem_load.partition_D(cS)
        tScS_t2r = split_wg(tScS_t2r, 2, wg_idx)
        base_row = m_block * self.kBlockM
        base_col = n_block * self.kBlockN
        row_id, col_id = (1, 0) if cutlass.const_expr(self.swapAB) else (0, 1)

        for i in cutlass.range(cute.size(preds), unroll_full=True):
            block_row = cute.get(tScS_t2r[i], mode=[row_id])
            block_col = cute.get(tScS_t2r[i], mode=[col_id])
            preds[i] = Boolean(block_row + base_row >= 0 and block_row + base_row < self.seqlen_q and block_col + base_col < self.seqlen_k)
