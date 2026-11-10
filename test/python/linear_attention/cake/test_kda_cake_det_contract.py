# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject contracts that need a different kernel chain."""

from types import SimpleNamespace
import cudnn
import pytest
from cudnn.linear_attention.cake.kda_det_engine import check_contract
from linear_attention.cake.test_gdn_cake_contract import facts

pytestmark = [pytest.mark.L0]


@pytest.mark.parametrize(
    "changes",
    [
        dict(h_q=8, h_k=8, h_v=8),
        dict(h_v=32),
        dict(use_qk_l2norm=True),
        dict(safe_gate=True),
        dict(n_seq=65537),
        dict(checkpoint_every_n_tokens=64),
        dict(beta_dtype=cudnn.data_type.FLOAT),
        dict(gate_channels=0),
    ],
)
def test_unsupported(changes):
    config = dict(op="KDA", h_q=16, h_k=16, h_v=16, gate_channels=128, beta_dtype=cudnn.data_type.BFLOAT16, use_qk_l2norm=False)
    with pytest.raises(NotImplementedError, match="kda_cake_det"):
        check_contract(facts(**(config | changes)), SimpleNamespace(inputs={}, outputs={}))
