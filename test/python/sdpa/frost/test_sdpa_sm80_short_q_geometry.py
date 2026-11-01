# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM80 short dense D128 queries use a smaller CTA without overriding knobs."""

import pytest
import torch

from cudnn.api_base import TensorDesc
from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm80
from cudnn.sdpa.fwd.config_sm80 import TemplateParams, validate_params

pytestmark = pytest.mark.L0


@pytest.mark.parametrize(
    "sq,d,dv,thd,tile_m,expected",
    [
        (1, 128, 128, False, None, (64, 4)),
        (8, 128, 128, False, None, (64, 4)),
        (64, 128, 128, False, None, (64, 4)),
        (65, 128, 128, False, None, (128, 8)),
        (129, 128, 128, False, None, (128, 8)),
        (8, 128, 128, False, 128, (128, 8)),
        (129, 128, 128, False, 64, (64, 4)),
        (8, 96, 80, False, None, (128, 8)),
        (8, 96, 80, False, 64, (64, 4)),
        (8, 64, 64, False, None, (128, 4)),
        (8, 192, 128, False, None, (128, 8)),
        (8, 256, 256, False, None, (128, 8)),
        (8, 128, 128, True, None, (128, 8)),
        (8, 128, 128, True, 64, (64, 4)),
    ],
)
def test_declared_geometry_and_explicit_tile(sq, d, dv, thd, tile_m, expected, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    descs = {}
    for role, h, s, width in (("q", 8, sq, d), ("k", 2, 257, d), ("v", 2, 257, dv), ("o", 8, sq, dv)):
        descs[role] = TensorDesc(
            dtype=torch.bfloat16,
            shape=(2, h, s, width),
            stride=(s * h * width, width, h * width, 1),
            stride_order=(3, 1, 2, 0),
            device=torch.device("cuda:0"),
            name=role,
        )
    api = SdpaFwdDslSm80(
        **{"sample_" + role: desc for role, desc in descs.items()},
        tile_m=tile_m,
        thd=thd,
        seq_kv_lens_present=thd,
        max_total_seq_len_q=2 * sq if thd else None,
        max_total_seq_len_kv=514 if thd else None,
    )
    assert api.check_support()
    assert (api.kernel_tile_m, api.kernel_num_warps) == expected
    validate_params(
        TemplateParams(
            d_qk=api.flavor_d_qk,
            d_v=api.flavor_d_v,
            tile_m=api.kernel_tile_m,
            num_warps=api.kernel_num_warps,
            tile_n=api.kernel_tile_n,
        )
    )
