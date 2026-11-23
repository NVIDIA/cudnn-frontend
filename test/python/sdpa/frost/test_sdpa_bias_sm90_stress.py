# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warm refill coverage for the SM90 staged additive bias: the D512 body (engine 20540) and the dedicated D128 template
(engine 20541).

On the D512 body each WG1 warp refills its shared bias rows with cp.async once per KV tile, after its reads of the
previous tile; on the D128 template WG0 refills each compute warpgroup's bias slot by TMA once per KV tile, after that
warpgroup's tile_dsl.consume release of its reads (python/cudnn/AGENTS.md Rule 11). This drives both refills under warm,
back-to-back launches.
"""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 90, reason="requires native Hopper")]


@pytest.mark.parametrize(
    "dtype,bias_fp32,tile,persistent",
    [
        (torch.float16, True, 64, False),
        (torch.bfloat16, False, 64, False),
        (torch.float16, False, 128, True),
        (torch.bfloat16, False, 128, True),
        (torch.bfloat16, False, 128, False),
    ],
    ids=["d512-fp16-fp32-bias", "d512-bf16-io-bias", "d128-fp16-io-bias", "d128-bf16-io-bias", "d128-bf16-io-bias-one-tile"],
)
def test_sm90_bias_stage_refill_stress(dtype, bias_fp32, tile, persistent, monkeypatch):
    """The D128 template's shipped persistent record runs one CTA per SM, below the stress control's geometry, so its
    one-tile-per-CTA record (4.4 thread blocks per SM here) runs the same slot refill at a geometry the control
    qualifies."""
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90
    from warm_stress import StressCase, assert_warm_stress

    if tile == 128 and not persistent:
        monkeypatch.setattr(api_dsl, "D128_PERSISTENT", False)

    gen = torch.Generator().manual_seed(2193)
    b, h, hk, sq, sk, d = 3, 64, 16, 257, 513, 128
    q = (torch.randn(b, h, sq, d, generator=gen) * 0.2).to(dtype).cuda()
    k = (torch.randn(b, hk, sk, d, generator=gen) * 0.2).to(dtype).cuda()
    v = (torch.randn(b, hk, sk, d, generator=gen) * 0.2).to(dtype).cuda()
    bias = (torch.randn(b, h, sq, 520, generator=gen) * 0.7).to(torch.float32 if bias_fp32 else dtype).cuda()[..., :sk]
    o = torch.empty_like(q)
    lse = torch.empty(b, h, sq, dtype=torch.float32, device="cuda")
    api = SdpaFwdDslSm90(q, k, v, o, sample_lse=lse, sample_bias=bias, pack_gqa=False, scale_softmax=d**-0.5, tile_m=tile, tile_n=tile)
    api.check_support()
    api.compile()
    assert api._dense_spec.has_bias
    kernel = "/sm90/d128_f16_multi_q/kernel.py" if tile == 128 else "/sm90/f16_three_wg/kernel.py"
    assert api._k_mod.__file__.replace("\\", "/").endswith(kernel)

    def launch():
        api.execute(q, k, v, o, lse, bias_tensor=bias)

    launch()
    # Toleranced correctness is separate from the stress helper's cold byte reference.
    for batch in range(b):
        scores = (q[batch].double() @ k[batch].double().repeat_interleave(4, 0).transpose(-1, -2)) * d**-0.5 + bias[batch].double()
        ref = scores.softmax(-1) @ v[batch].double().repeat_interleave(4, 0)
        torch.testing.assert_close(o[batch].double(), ref, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(lse[batch].double(), scores.logsumexp(-1), atol=3e-4, rtol=3e-4)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    items = b * h * ((sq + tile - 1) // tile)
    kv_tiles = (sk + tile - 1) // tile
    if persistent:
        # The D128 record the adapter ships here is persistent: one CTA per SM, each walking items // SMs or more items,
        # and the slots' refills carry across them.
        assert api.params.persistent and items >= 2 * sms
        blocks, refills = min(items, sms), kv_tiles * (items // sms) - 1
    else:
        assert not api.params.persistent
        assert items >= 2 * sms
        blocks, refills = items, kv_tiles - 1
    assert_warm_stress(
        StressCase(
            name=f"sm90-{'d128' if tile == 128 else 'three-wg'}-bias-{dtype}-{'fp32' if bias_fp32 else 'io'}-bias{'' if persistent or tile == 64 else '-one-tile'}",
            launch=launch,
            outputs={"o": o, "lse": lse},
            thread_blocks=blocks,
            refills_per_launch=refills,
        ),
    )
