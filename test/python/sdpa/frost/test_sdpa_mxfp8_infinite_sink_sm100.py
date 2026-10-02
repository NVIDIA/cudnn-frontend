# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact sink limits across the SM100 MXFP8 template families."""

import math
import pytest
import torch
from frost_test_utils import requires_pre_rubin_blackwell, requires_dsl
from sdpa.frost.test_sdpa_fwd_mxfp8_sm100 import _run

pytestmark = [pytest.mark.L0, requires_pre_rubin_blackwell, requires_dsl]


@pytest.mark.parametrize(
    "d,dv,in_key,out_dtype",
    [
        (128, 128, "e4m3", torch.float16),
        (128, 128, "e5m2", torch.float8_e5m2),
        (192, 128, "e4m3", torch.bfloat16),
        (192, 128, "e5m2", torch.float8_e4m3fn),
        (256, 256, "e4m3", torch.float8_e4m3fn),
        (256, 256, "e5m2", torch.float16),
        (512, 512, "e4m3", torch.float8_e5m2),
        (512, 512, "e5m2", torch.bfloat16),
    ],
)
def test_mxfp8_infinite_sink_zeroes_output_and_preserves_query_trim(d, dv, in_key, out_dtype):
    q_lens = [0, 129, 256]
    kv_lens = [256, 0, 137]
    for value in (1000.0, float("inf")):
        # A large finite sink is the control; +inf has the exact same zero-O limit.
        torch.manual_seed(1)
        sink = torch.full((1, 4, 1, 1), value, device="cuda")
        result = _run(
            3,
            4,
            2,
            256,
            in_key,
            out_dtype,
            scale=1.0 / math.sqrt(d),
            sdpa_kwargs={"use_causal_mask_bottom_right": True},
            sink=sink,
            seq_lens_q=q_lens,
            seq_lens_kv=kv_lens,
            d_qk=d,
            d_v=dv,
            return_lse=True,
            poison_tmem_before_execute=True,
        )
        assert (result.output.float() == 0).all(), f"sink={value}: O must be exactly zero"
        assert result.amax.item() == 0.0
        expected = torch.full_like(result.stats, value)
        for batch, length in enumerate(q_lens):
            expected[batch, :, length:] = float("-inf")
        torch.testing.assert_close(result.stats, expected, atol=1e-3, rtol=0)


@pytest.mark.parametrize("block_scaled_o", ["mxfp8", "nvfp4"])
def test_mxfp8_infinite_sink_block_scaled_output(block_scaled_o):
    result = _run(
        1,
        4,
        2,
        256,
        "e4m3",
        torch.float8_e4m3fn,
        scale=1.0 / math.sqrt(128),
        sdpa_kwargs={},
        sink=torch.full((1, 4, 1, 1), float("inf"), device="cuda"),
        block_scaled_o=block_scaled_o,
        scale_o=3.0 if block_scaled_o == "nvfp4" else None,
    )
    assert (result.output == 0).all()
    assert result.amax == 0.0
    assert result.sf_pad_ok
