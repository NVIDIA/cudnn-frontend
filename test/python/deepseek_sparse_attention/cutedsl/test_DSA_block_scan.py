# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import pytest
import torch

from cudnn.deepseek_sparse_attention.indexer_top_k.block_scan import block_prefix_sum


@cute.jit
def _launch_scan(x: cute.Tensor, y: cute.Tensor, size: cutlass.Constexpr):
    block_prefix_sum(size, 128, x, y).launch(grid=(1, 1, 1), block=(128, 1, 1))


@pytest.mark.L0
@pytest.mark.parametrize("size", [128, 1024], ids=["one-tile", "eight-tiles"])
def test_block_scan_reuses_warp_sums(size):
    x = torch.arange(size, device="cuda", dtype=torch.int32) % 7
    y = torch.empty_like(x)
    expected = x.cumsum(dim=0, dtype=torch.int32)
    a, b = from_dlpack(x), from_dlpack(y)
    run = cute.compile(_launch_scan, a, b, size)
    for _ in range(20):
        run(a, b)
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
