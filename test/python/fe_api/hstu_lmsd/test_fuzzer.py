# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shape and dropout fuzz coverage for HSTU LMSD forward."""

import pytest
import torch

try:
    import cutlass  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(f"CuTe DSL is unavailable: {exc}", allow_module_level=True)

from cudnn.hstu_lmsd import hstu_lmsd_forward

pytestmark = [
    pytest.mark.gpu_exclusive,
    pytest.mark.xdist_group(name="gpu_exclusive"),
]

_IS_SM10X = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
@pytest.mark.parametrize(
    "n,dropout_ratio,seed",
    ((1, 0.0, 0), (37, 0.1, 7), (1025, 0.35, 2**40 + 9)),
)
def test_forward_matches_mask_reconstruction(n, dropout_ratio, seed):
    torch.manual_seed(11 + n)
    d = 512
    x = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
    u_storage = torch.randn((n, 4 * d), device="cuda", dtype=torch.bfloat16)
    u = u_storage[:, :d]
    weight = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((d,), device="cuda", dtype=torch.bfloat16)

    eps = 1e-6
    result = hstu_lmsd_forward(
        x,
        u,
        weight,
        bias,
        eps=eps,
        dropout_ratio=dropout_ratio,
        seed=seed,
    )
    y, mean, rstd, mask = result
    mask_i32 = mask.to(torch.int32)
    assert torch.count_nonzero(mask_i32 & ~0x7) == 0

    xf = x.float()
    uf = u.float()
    expected_mean = xf.mean(dim=1)
    expected_var = (xf.square().mean(dim=1) - expected_mean.square()).clamp_min(0.0)
    expected_rstd = torch.rsqrt(expected_var + eps)
    torch.testing.assert_close(mean, expected_mean, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(rstd, expected_rstd, rtol=2e-4, atol=2e-4)

    normalized = (xf - expected_mean[:, None]) * expected_rstd[:, None]
    ln = normalized * weight.float() + bias.float()
    silu = torch.nn.functional.silu(uf)
    scale = 1.0 / (1.0 - dropout_ratio)
    zero = torch.zeros((), device=x.device)
    expected = torch.cat(
        (
            torch.where((mask_i32 & 4) != 0, silu * scale, zero),
            torch.where((mask_i32 & 2) != 0, xf * scale, zero),
            torch.where((mask_i32 & 1) != 0, ln * silu * scale, zero),
        ),
        dim=1,
    )
    torch.testing.assert_close(y.float(), expected, rtol=1.5e-2, atol=1.5e-2)
