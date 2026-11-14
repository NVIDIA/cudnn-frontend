# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warm stage-refill coverage for dense D128 additive-bias forward."""

import pytest
import torch

from frost_test_utils import requires_blackwell_geforce, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_sm120_bias_stage_refill_stress(dtype):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120
    from warm_stress import StressCase, assert_warm_stress

    gen = torch.Generator().manual_seed(2193)
    b, h, hk, sq, sk, d = 3, 64, 16, 257, 513, 128
    q = (torch.randn(b, h, sq, d, generator=gen) * 0.2).to(dtype).cuda()
    k = (torch.randn(b, hk, sk, d, generator=gen) * 0.2).to(dtype).cuda()
    v = (torch.randn(b, hk, sk, d, generator=gen) * 0.2).to(dtype).cuda()
    bias = (torch.randn(b, h, sq, 520, generator=gen) * 0.7).cuda()[..., :sk]
    o = torch.empty_like(q)
    lse = torch.empty(b, h, sq, dtype=torch.float32, device="cuda")
    api = SdpaFwdDslSm120(q, k, v, o, sample_lse=lse, sample_bias=bias, tile_m=128, tile_n=128, pack_gqa=False, scale_softmax=d**-0.5)
    api.check_support()
    api.compile()
    assert api._dense_spec.has_bias
    assert api._k_mod.__file__.replace("\\", "/").endswith("/sm120/prefill_f16.py")

    def launch():
        api.execute(q, k, v, o, lse, bias_tensor=bias)

    launch()
    # Toleranced correctness is separate from the stress helper's cold byte reference.
    for batch in range(b):
        scores = (q[batch].double() @ k[batch].double().repeat_interleave(4, 0).transpose(-1, -2)) * d**-0.5 + bias[batch].double()
        ref = scores.softmax(-1) @ v[batch].double().repeat_interleave(4, 0)
        torch.testing.assert_close(o[batch].double(), ref, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(lse[batch].double(), scores.logsumexp(-1), atol=3e-4, rtol=3e-4)
    blocks = b * h * ((sq + 127) // 128)
    assert blocks >= 2 * torch.cuda.get_device_properties(0).multi_processor_count
    assert_warm_stress(
        StressCase(
            name=f"sm120-d128-bias-{dtype}-m128n128",
            launch=launch,
            outputs={"o": o, "lse": lse},
            thread_blocks=blocks,
            refills_per_launch=(sk + 127) // 128 - 1,
        ),
    )
