# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warm refill coverage for the SM90 D512 body's staged additive bias (engine 20540).

Each WG1 warp refills its shared bias rows with cp.async once per KV tile, after its reads of the previous tile
(python/cudnn/AGENTS.md Rule 11); this drives that refill under warm, back-to-back launches.
"""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 90, reason="requires native Hopper")]


@pytest.mark.parametrize("dtype,bias_fp32", [(torch.float16, True), (torch.bfloat16, False)], ids=["fp16-fp32-bias", "bf16-io-bias"])
def test_sm90_bias_stage_refill_stress(dtype, bias_fp32):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90
    from warm_stress import StressCase, assert_warm_stress

    gen = torch.Generator().manual_seed(2193)
    b, h, hk, sq, sk, d = 3, 64, 16, 257, 513, 128
    q = (torch.randn(b, h, sq, d, generator=gen) * 0.2).to(dtype).cuda()
    k = (torch.randn(b, hk, sk, d, generator=gen) * 0.2).to(dtype).cuda()
    v = (torch.randn(b, hk, sk, d, generator=gen) * 0.2).to(dtype).cuda()
    bias = (torch.randn(b, h, sq, 520, generator=gen) * 0.7).to(torch.float32 if bias_fp32 else dtype).cuda()[..., :sk]
    o = torch.empty_like(q)
    lse = torch.empty(b, h, sq, dtype=torch.float32, device="cuda")
    api = SdpaFwdDslSm90(q, k, v, o, sample_lse=lse, sample_bias=bias, pack_gqa=False, scale_softmax=d**-0.5)
    api.check_support()
    api.compile()
    assert api._dense_spec.has_bias
    assert api._k_mod.__file__.replace("\\", "/").endswith("/sm90/f16_three_wg/kernel.py")

    def launch():
        api.execute(q, k, v, o, lse, bias_tensor=bias)

    launch()
    # Toleranced correctness is separate from the stress helper's cold byte reference.
    for batch in range(b):
        scores = (q[batch].double() @ k[batch].double().repeat_interleave(4, 0).transpose(-1, -2)) * d**-0.5 + bias[batch].double()
        ref = scores.softmax(-1) @ v[batch].double().repeat_interleave(4, 0)
        torch.testing.assert_close(o[batch].double(), ref, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(lse[batch].double(), scores.logsumexp(-1), atol=3e-4, rtol=3e-4)
    blocks = b * h * ((sq + 63) // 64)
    assert blocks >= 2 * torch.cuda.get_device_properties(0).multi_processor_count
    assert_warm_stress(
        StressCase(
            name=f"sm90-three-wg-bias-{dtype}-{'fp32' if bias_fp32 else 'io'}-bias",
            launch=launch,
            outputs={"o": o, "lse": lse},
            thread_blocks=blocks,
            refills_per_launch=(sk + 63) // 64 - 1,
        ),
    )
