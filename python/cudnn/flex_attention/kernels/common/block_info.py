# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
from dataclasses import dataclass
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from cudnn.flex_attention.kernels.common.seqlen_info import SeqlenInfoQK


@dataclass(frozen=True)
class BlockInfo:
    """Dense sequence bounds used around planner-provided sparse block lists."""

    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]

    @cute.jit
    def get_n_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
    ) -> Tuple[Int32, Int32]:
        del m_block
        return Int32(0), cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)

    @cute.jit
    def get_m_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        n_block: Int32,
    ) -> Tuple[Int32, Int32]:
        del n_block
        return Int32(0), cute.ceil_div(seqlen_info.seqlen_q, self.tile_m)

    @cute.jit
    def get_n_block_max_for_m_block(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
    ) -> Int32:
        del m_block
        return cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)
