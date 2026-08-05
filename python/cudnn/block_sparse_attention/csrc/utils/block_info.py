# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SPDX-License-Identifier: MIT
from typing import Tuple, Optional
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

from cudnn.block_sparse_attention.csrc.utils.seqlen_info import SeqlenInfoQK


@dataclass(frozen=True)
class BlockInfo:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    is_causal: cutlass.Constexpr[bool]
    is_local: cutlass.Constexpr[bool] = False
    is_split_kv: cutlass.Constexpr[bool] = False
    window_size_left: Optional[Int32] = None
    window_size_right: Optional[Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1

    @cute.jit
    def get_m_block_min_max(self, seqlen_info: SeqlenInfoQK, n_block: Int32) -> Tuple[Int32, Int32]:
        m_block_max = cute.ceil_div(seqlen_info.seqlen_q, self.tile_m)
        m_block_min = 0
        if const_expr(self.is_causal or (self.is_local and self.window_size_right is not None)):
            n_idx_min = n_block * self.tile_n
            m_idx = n_idx_min + seqlen_info.seqlen_q - seqlen_info.seqlen_k
            m_idx_right = m_idx if const_expr(self.is_causal) else m_idx - self.window_size_right
            m_block_min = max(m_block_min, m_idx_right // self.tile_m)
        if const_expr(self.is_local and self.window_size_left is not None):
            n_idx_max = (n_block + 1) * self.tile_n
            m_idx = n_idx_max + seqlen_info.seqlen_q - seqlen_info.seqlen_k
            m_idx_left = m_idx + self.window_size_left
            m_block_max = min(m_block_max, cute.ceil_div(m_idx_left, self.tile_m))
        return m_block_min, m_block_max
