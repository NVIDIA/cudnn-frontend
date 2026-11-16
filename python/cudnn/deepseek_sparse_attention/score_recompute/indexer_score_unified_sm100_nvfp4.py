# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 CuTeDSL unified NVFP4 dense indexer-score kernel.

NVFP4 Q/K are packed E2M1 (two values per byte), the data format of MXFP4,
with one E4M3 scale per 16 head_dim elements instead of one E8M0 scale per 32.
The kernel runs the MXFP4 producer, scale path and unified indexer epilogue on
tcgen05 ``kind::mxf4nvf4`` with 16-value scale blocks, whose MMA reads K=64
values and four E4M3 scales (one TMEM column) per row.  A row's eight scales
span two TMEM columns of its scale field, and each MMA along K reads its own.
Per-tensor global scales are not applied: callers fold them into
``softmax_scale``.
"""

from __future__ import annotations

import cutlass

from .indexer_score_unified_sm100_mxfp4 import IndexerScoreUnifiedSm100Mxfp4


class IndexerScoreUnifiedSm100Nvfp4(IndexerScoreUnifiedSm100Mxfp4):
    """NVFP4 dense score kernel with unified indexer output semantics."""

    format_name = "NVFP4"
    sf_vec_size = 16
    scale_dtype = cutlass.Float8E4M3FN
