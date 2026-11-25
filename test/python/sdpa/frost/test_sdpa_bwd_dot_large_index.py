# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Large FP32 delta indexing with owned storage on either side of the pointer."""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl
from test_sdpa_bwd_dot_grid_determinism import _compile_dense

pytestmark = [pytest.mark.L1, requires_dsl, pytest.mark.skipif(_SM is None, reason="CUDA device required")]


def test_delta_indices_cross_signed_int32():
    # The old Int32 delta offset wraps into the owned prefix, never outside
    # this allocation. Keep this large-storage detector out of smoke CI.
    free, _ = torch.cuda.mem_get_info()
    if free < 28 * 1024**3:
        pytest.skip("Needs 28 GiB free for delta, owned underflow guard and inputs")
    batch, heads, rounded = 65536, 257, 128
    prefix, count = 2**31, batch * heads * rounded
    operands = [torch.ones((batch, 1, heads, 64), dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    raw = torch.full((prefix + count + 16,), 12345.0, dtype=torch.float32, device="cuda")
    delta = raw[prefix : prefix + count].view(batch, heads, rounded)
    fn = _compile_dense(operands, delta)
    fn(*operands, delta, None, None, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert torch.all(delta[:, :, 0] == 64.0), "live delta rows cross the signed Int32 boundary"
    assert torch.count_nonzero(delta[:, :, 1:]) == 0
    assert torch.all(raw[:prefix] == 12345.0), "wrapped delta stores changed the owned underflow guard"
    assert torch.all(raw[-16:] == 12345.0)
