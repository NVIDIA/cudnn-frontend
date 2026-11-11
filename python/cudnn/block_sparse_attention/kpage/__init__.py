# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KPage: page-compacted execution of SM100/SM103 blk128 block-sparse attention.

Selected K/V blocks are split into 16-row pages and the valid pages of
different blocks are gathered into the same 128-row MMA tile during the TMA
load, so a partially occupied block costs at most 15 padded rows instead of
up to 127.  Query blocks with at most 64 valid rows run on an M64 x N256
kernel, the rest on M128 x N128.  Backward packs K/V pages into owner tiles
and walks the union of their Q pages with a per-step page-pair mask.

Enabled through ``kpage=True`` (or a prepared :class:`KPageLayout`) on
``block_sparse_attention_forward`` / ``block_sparse_attention_backward``.
"""

from ._layout import KPageLayout, prepare_layout

__all__ = ["KPageLayout", "prepare_layout"]
