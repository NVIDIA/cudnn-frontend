# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic scheduler, multi-CTA cluster: a cluster's CTAs must not exit while a peer still releases the last claim.

Every CTA releases each claimed cluster tile with a remote arrive on the leader CTA's shared memory.  One token over
many experts gives a single tile: most clusters claim nothing, and every CTA walks all the expert offsets before it
releases the empty claim, so a leader that exits after its own release lets a slower peer arrive into an exited CTA
(an unspecified launch failure).  1024 experts and an L2 flush before each launch (the offset loads miss) make the
CTAs' walks long and uneven.
"""

from __future__ import annotations

import pytest
import torch

EXPERTS = 1024
LAUNCHES = 2000


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from cudnn.tensor_adapter import has_tcgen05

    if not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("SM100 is required")


@pytest.mark.L0
@pytest.mark.parametrize(
    ("mma_tiler_mn", "cluster_shape_mn"),
    [((256, 256), (2, 2)), ((128, 256), (1, 2)), ((128, 256), (2, 1))],
    ids=["2cta-cluster-2x2", "1cta-cluster-1x2", "1cta-cluster-2x1"],
)
def test_grouped_gemm_bf16_dynamic_sched_one_token_many_empty_experts(mma_tiler_mn, cluster_shape_mn):
    from cudnn import grouped_gemm_wrapper_sm100

    m, n, k = 256, 1024, 512
    g = torch.Generator(device="cuda").manual_seed(0)
    a = (torch.randn((1, m, k), generator=g, device="cuda", dtype=torch.bfloat16) * 0.125).permute(1, 2, 0)
    b = (torch.randn((EXPERTS, n, k), generator=g, device="cuda", dtype=torch.bfloat16) * 0.125).permute(1, 2, 0)
    alpha = torch.ones(EXPERTS, dtype=torch.float32, device="cuda")
    prob = torch.ones((m, 1, 1), dtype=torch.float32, device="cuda")
    offsets = torch.empty(EXPERTS, dtype=torch.int32, device="cuda")
    flush = torch.empty(128 << 20, dtype=torch.float32, device="cuda")  # 512 MB, larger than the L2
    picks = torch.randint(EXPERTS, (LAUNCHES,), generator=torch.Generator().manual_seed(0)).tolist()

    for i, expert in enumerate(picks):
        # One token, held by `expert` (padded to 256 rows); every other expert is empty.
        offsets.copy_((torch.arange(EXPERTS, dtype=torch.int32) >= expert).to(torch.int32) * m)
        flush.fill_(float(i))
        result = grouped_gemm_wrapper_sm100(
            a_tensor=a,
            b_tensor=b,
            padded_offsets=offsets,
            alpha_tensor=alpha,
            prob_tensor=prob,
            d_dtype=torch.bfloat16,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_dynamic_sched=True,
        )
        torch.cuda.synchronize()

    expected = (a[:, :, 0].float() @ b[:, :, picks[-1]].float().t()).to(torch.bfloat16)
    torch.testing.assert_close(result["d_tensor"][:, :, 0], expected, rtol=2e-2, atol=2e-2)
