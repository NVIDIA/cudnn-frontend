# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch-local sequence metadata for HSTU attention kernels."""

from typing import Optional

import cutlass
import cutlass.cute as cute


class SeqlenInfo:
    """Load and cache one packed sequence's offsets and lengths."""

    def __init__(
        self,
        batch_idx: cutlass.Int32,
        max_seqlen_q: cutlass.Int32,
        max_seqlen_k: cutlass.Int32,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        page_indptrs: Optional[cute.Tensor] = None,
        tile_m: int = 128,
    ):
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise AssertionError("packed HSTU attention requires Q and K sequence offsets")

        next_batch_idx = batch_idx + 1
        q_start = cu_seqlens_q[batch_idx]
        k_start = cu_seqlens_k[batch_idx]
        q_end = cu_seqlens_q[next_batch_idx]
        k_end = cu_seqlens_k[next_batch_idx]

        self.offset_q = q_start
        self.offset_k = k_start
        self.seqlen_q = q_end - q_start
        self.seqlen_k = k_end - k_start

        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k

        padded_q_block = q_start // tile_m + batch_idx
        self.padded_offset_q = cute.assume(padded_q_block * tile_m, divby=tile_m)

        if page_indptrs is None:
            self.page_ind = 0
        else:
            self.page_ind = page_indptrs[batch_idx]
