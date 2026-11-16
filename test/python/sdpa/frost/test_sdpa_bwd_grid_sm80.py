# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native boundary controls for the SM80 backward grid admission check."""

import pytest
import torch

from frost_test_utils import requires_dsl

pytestmark = [
    pytest.mark.L1,
    requires_dsl,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0), reason="requires SM80"),
]


@pytest.mark.parametrize("axis", ["batch", "heads"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_grid_boundary(axis, dtype):
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80
    from cudnn.frost.tile_dsl.constants import SCHED_LPT_L2, SCHED_NATURAL

    for extent, causal in ((65535, False), (65536, False), (65536, True)):
        b, h = (extent, 1) if axis == "batch" else (1, extent)
        sq, sk, d = 2, 3, 8
        torch.manual_seed(1573)
        q, k, v, do = [torch.randn(b, h, s, d, device="cuda", dtype=dtype) * 0.2 for s in (sq, sk, sk, sq)]
        q64, k64, v64 = [x.double().requires_grad_() for x in (q, k, v)]
        scores = (q64 @ k64.transpose(-1, -2)) * d**-0.5
        if causal:
            scores = scores.masked_fill(torch.arange(sk, device="cuda")[None, :] > torch.arange(sq, device="cuda")[:, None], -torch.inf)
        stats = scores.logsumexp(-1, keepdim=True).float()
        o64 = scores.softmax(-1) @ v64
        references = torch.autograd.grad(o64, (q64, k64, v64), do.double())
        o = o64.detach().to(dtype)
        dq, dk, dv = [torch.full_like(x, torch.nan) for x in (q, k, v)]
        api = SdpaBwdDslSm80(
            sample_q=q,
            sample_k=k,
            sample_v=v,
            sample_o=o,
            sample_do=do,
            sample_stats=stats,
            sample_dq=dq,
            sample_dk=dk,
            sample_dv=dv,
            is_causal=causal,
        )
        if extent > 65535 and not causal:
            with pytest.raises(NotImplementedError, match="grid.*65535"):
                api.check_support()
            assert all(torch.isnan(x).all() for x in (dq, dk, dv))
            continue
        assert api.check_support()
        api.compile()
        assert api._params.sched_policy == (SCHED_LPT_L2 if causal else SCHED_NATURAL)
        workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
        api.execute(q, k, v, o, do, stats, dq, dk, dv, workspace=workspace)
        for actual, reference in zip((dq, dk, dv), references):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual.double(), reference, atol=2e-4 if dtype == torch.float16 else 2e-3, rtol=0.03)
