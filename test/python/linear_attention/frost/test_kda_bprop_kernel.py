# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST KDA backward: currently a STUB on this branch (the small-chunk
design recomputes the forward states in the backward once the kernel lands).
The contract: ``KdaFrostEngine`` declines ``KDA_BWD``
graphs so the router can fall back."""

from __future__ import annotations

import pytest

import cudnn  # noqa: F401  (conftest extends cudnn.__path__ with the source tree)

from linear_attention.common import assert_engine_declines

pytestmark = pytest.mark.L0


def test_kda_bwd_frost_engine_declines():

    total, H, D = 256, 2, 128
    g = cudnn.pygraph()
    q_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="q")
    k_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="k")
    v_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="v")
    g_t = g.tensor([total, H, D], data_type=cudnn.data_type.FLOAT, name="g")
    beta_t = g.tensor([total, H], data_type=cudnn.data_type.FLOAT, name="beta")
    cu_t = g.tensor([2], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    dO_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="dO")
    g.kda_bwd(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu_seqlens=cu_t,
        dO=dO_t,
        scale=0.125,
        name="kda_bwd",
    )
    assert_engine_declines(g, "kda_frost")  # stub backward kernel on this branch
