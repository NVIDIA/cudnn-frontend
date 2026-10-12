# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph admission and gate binding use the same TMA layout contract."""

import pytest

from cudnn.sdpa.graph_analyzer import gate_layout_ok

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("elem_bytes", [1, 2, 4])
@pytest.mark.parametrize("batch", [1, 2])
def test_gate_batch_stride_alignment(elem_bytes, batch):
    shape = (batch, 4, 17, 256)
    compact = (17 * 4 * 256, 256, 4 * 256, 1)
    assert gate_layout_ok(shape, compact, elem_bytes)
    assert gate_layout_ok(shape, (compact[0] + 16, *compact[1:]), elem_bytes)
    assert gate_layout_ok(shape, (compact[0] + 1, *compact[1:]), elem_bytes) == (batch == 1)
