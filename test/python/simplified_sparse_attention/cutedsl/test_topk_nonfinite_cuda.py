# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Top-k ids stay inside each row's key bounds when indexer keys hold NaN/Inf (SM103).

The vendored DKG top-k's float-compare classifier (``boundary_classify``)
dropped non-finite and exact +0.0 K-th scores and left output slots
unwritten. SSA disables it on its exact top-k calls, so every value follows
the integer key order. Output buffers start as a sentinel, so an unwritten
slot fails the bounds check.
"""

import sys

import pytest

import torch

from cudnn.simplified_sparse_attention.cutedsl import backend

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
        reason="The packaged CuTe kernels require SM100 or SM103",
    ),
]


SENTINEL = 0x7FFFFFF0


@pytest.mark.parametrize("tk", [4096, 131072], ids=["dense", "filtered"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "posinf", "neginf"])
def test_nonfinite_keys_keep_ids_in_bounds(tk, value):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(512)
    tq, topk = 256, 512
    q_idx = (0.2 * torch.randn((tq, 128), device=device, generator=generator)).bfloat16()
    k_idx = (0.2 * torch.randn((tk, 128), device=device, generator=generator)).bfloat16()
    # Whole non-finite rows make the affected scores NaN or +-inf.
    k_idx[torch.randperm(tk, device=device, generator=generator)[: tk // 100]] = value
    rows = torch.arange(tq, device=device)
    valid = rows % 5 != 0
    starts = torch.where(rows % 2 == 0, 0, tk // 2).to(torch.int32)
    ends = torch.where(valid, tk, starts).to(torch.int32)
    indices = torch.full((tq, topk), SENTINEL, dtype=torch.int32, device=device)
    predict = torch.empty((tq, topk), dtype=torch.float32, device=device)
    fused = sys.modules.get("cudnn.simplified_sparse_attention.cutedsl.kernels.indexer_forward_topk_fused_tcgen05")
    for state in getattr(fused, "_state_filtered", {}).values():
        state["out_idx"].fill_(SENTINEL)

    backend._kernel_module("indexer_forward_topk").run(
        q_idx,
        k_idx,
        indices,
        predict,
        starts,
        ends,
        triangular=False,
        score_impl="auto",
        fusion="auto",
        topk_impl="auto",
        topk_overflow_policy="REREAD_ALWAYS",
    )
    torch.cuda.synchronize()

    ids = indices.long()
    inside = (ids >= starts[:, None].long()) & (ids < ends[:, None].long())
    bad_rows = int((~inside[valid]).any(dim=1).sum())
    assert bad_rows == 0, f"{bad_rows} valid rows hold ids outside their key bounds"
