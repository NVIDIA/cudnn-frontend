# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WorkspaceCarver (recipe R2): the caller's buffer is carved, never copied."""

import pytest
import torch

from cudnn.api_base import _WS_ALIGN, WorkspaceCarver


@pytest.mark.L0
def test_workspace_carver_rejects_strided_and_misaligned_buffers():
    """A strided view is refused up front (reshape would copy it: a hidden allocation carved into
    instead of the caller's buffer); the default base alignment is 16 bytes and a TMA-descriptor
    consumer asks for _WS_ALIGN; carving allocates nothing."""
    base = torch.empty(8192, dtype=torch.uint8, device="cuda")
    assert WorkspaceCarver(base, 1024, "probe").take(256, torch.float32).numel() == 256

    strided = torch.empty(16384, dtype=torch.uint8, device="cuda")[::2]
    with pytest.raises(ValueError, match="must be contiguous"):
        WorkspaceCarver(strided, 1024, "probe")

    off16 = base[16:]  # torch storage bases are 512-byte aligned: this one is 16-byte but not 128-byte aligned
    WorkspaceCarver(off16, 1024, "probe")
    with pytest.raises(ValueError, match=f"{_WS_ALIGN}-byte aligned"):
        WorkspaceCarver(off16, 1024, "probe", align=_WS_ALIGN)

    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    carver = WorkspaceCarver(base, 2048, "probe")
    carver.take(64, torch.int64)
    carver.take(100, torch.float32)
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
