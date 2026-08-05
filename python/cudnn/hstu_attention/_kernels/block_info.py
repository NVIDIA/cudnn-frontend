# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SPDX-License-Identifier: MIT

from typing import Tuple, Optional
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import const_expr

from .seqlen_info import SeqlenInfo


@dataclass(frozen=True)
class BlockInfo:
    cta_tiler: cutlass.Constexpr[Tuple[int, int]]
    is_causal: cutlass.Constexpr[bool] = False
    is_local: cutlass.Constexpr[bool] = False
    is_paged: cutlass.Constexpr[bool] = False
    window_size_left: Optional[cutlass.Int32] = None
    window_size_right: Optional[cutlass.Int32] = None

    @cute.jit
    def get_n_block_info(
        self, seqlen_info: SeqlenInfo, m_block: cutlass.Int32, offset_dynamic: cutlass.Int32
    ) -> Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
        seqlen_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q

        row_begin = m_block * self.cta_tiler[0] - offset_dynamic
        row_end = row_begin + self.cta_tiler[0]

        n_block_min = max(0, (row_begin + seqlen_offset - self.window_size_left) // self.cta_tiler[1]) if self.is_local else 0
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.cta_tiler[1])

        if self.is_causal or self.is_local:
            n_block_max = min(n_block_max, cute.ceil_div(row_end + seqlen_offset + self.window_size_right, self.cta_tiler[1]))

        n_masking_steps = 0
        if const_expr(not self.is_paged):
            n_masking_block_max = cute.ceil_div(min(seqlen_info.seqlen_k, row_end + seqlen_offset), self.cta_tiler[1])
            n_masking_block_min = (row_begin + seqlen_offset) // self.cta_tiler[1]

            # 1: first tile should be masked for boundary check
            n_masking_steps = 1 if not self.is_causal else n_masking_block_max - n_masking_block_min
        else:
            n_masking_pages = 0
            if row_begin + seqlen_offset < seqlen_info.seqlen_k:
                n_masking_pages = cute.ceil_div(min(seqlen_info.seqlen_k, row_end + seqlen_offset), self.cta_tiler[1])
                n_masking_pages -= max((row_begin + seqlen_offset) // self.cta_tiler[1], 0)
            n_masking_steps = n_masking_pages

        return n_block_max, n_block_min, n_masking_steps


@dataclass(frozen=True)
class BWDBlockInfo:
    cta_tiler: cutlass.Constexpr[Tuple[int, int]]
    is_causal: cutlass.Constexpr[bool] = False
    is_local: cutlass.Constexpr[bool] = False
    window_size_left: Optional[cutlass.Int32] = None
    window_size_right: Optional[cutlass.Int32] = None

    @cute.jit
    def get_m_block_info(
        self,
        seqlen_info: SeqlenInfo,
        n_block: cutlass.Int32,
    ) -> Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
        seqlen_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q

        m_masking_block_max = cute.ceil_div(max(0, (n_block + 1) * self.cta_tiler[1] - seqlen_offset), self.cta_tiler[0])

        m_masking_block_min = max(0, n_block * self.cta_tiler[1] - seqlen_offset) // self.cta_tiler[0]
        m_masking_steps = m_masking_block_max - m_masking_block_min if self.is_causal else 1

        m_block_min = (
            0
            if (not self.is_causal and not self.is_local)
            else max(0, (n_block * self.cta_tiler[1] - seqlen_offset - self.window_size_right) // self.cta_tiler[0])
        )
        m_block_max = cute.ceil_div(seqlen_info.seqlen_q, self.cta_tiler[0])

        if self.is_local:
            m_block_max = min(m_block_max, cute.ceil_div((n_block + 1) * self.cta_tiler[1] - seqlen_offset + self.window_size_left, self.cta_tiler[0]))

        return m_block_min, m_block_max, m_masking_steps
