# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal public-API usage tests for HSTU LMSD."""

import pytest
import torch

try:
    import cutlass  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(f"CuTe DSL is unavailable: {exc}", allow_module_level=True)

from cudnn.hstu_lmsd import hstu_lmsd_backward, hstu_lmsd_forward

pytestmark = [
    pytest.mark.gpu_exclusive,
    pytest.mark.xdist_group(name="gpu_exclusive"),
]

_IS_SM10X = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def _inputs(n: int = 257):
    torch.manual_seed(123)
    d = 512
    x = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
    u_storage = torch.randn((n, 4 * d), device="cuda", dtype=torch.bfloat16)
    u = u_storage[:, :d]
    weight = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
    return x, u, weight, bias


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
def test_explicit_forward_backward_usage():
    x, u, weight, bias = _inputs()
    forward = hstu_lmsd_forward(
        x,
        u,
        weight,
        bias,
        eps=1e-6,
        dropout_ratio=0.1,
        seed=17,
    )
    y, mean, rstd, mask = forward
    assert y.shape == (x.shape[0], 3 * x.shape[1])
    assert mean.shape == rstd.shape == (x.shape[0],)
    assert mask.shape == x.shape
    assert y.dtype == x.dtype
    assert mean.dtype == rstd.dtype == torch.float32
    assert mask.dtype == torch.int8

    dy = torch.randn_like(y)
    backward = hstu_lmsd_backward(
        dy,
        x,
        u,
        weight,
        bias,
        mean,
        rstd,
        mask,
        dropout_ratio=0.1,
    )
    dx, du, dweight, dbias = backward
    assert dx.shape == du.shape == x.shape
    assert dweight.shape == dbias.shape == weight.shape
    assert dx.dtype == du.dtype == dweight.dtype == dbias.dtype == x.dtype
    assert torch.isfinite(dx).all()
    assert torch.isfinite(du).all()
    assert torch.isfinite(dweight).all()
    assert torch.isfinite(dbias).all()
