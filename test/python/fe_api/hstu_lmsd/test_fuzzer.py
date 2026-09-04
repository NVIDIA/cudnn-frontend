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
from cudnn.hstu_lmsd.cutedsl._common import keep_threshold32

pytestmark = [
    pytest.mark.gpu_exclusive,
    pytest.mark.xdist_group(name="gpu_exclusive"),
]

_IS_SM10X = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10

_MASK32 = (1 << 32) - 1
_PHILOX_M0, _PHILOX_M1 = 0xD2511F53, 0xCD9E8D57
_PHILOX_W0, _PHILOX_W1 = 0x9E3779B9, 0xBB67AE85


def _philox4x32(counter0, counter1, counter2, counter3, key0, key1):
    for _ in range(10):
        product0 = _PHILOX_M0 * counter0
        product1 = _PHILOX_M1 * counter2
        counter0, counter1, counter2, counter3 = (
            ((product1 >> 32) ^ counter1 ^ key0) & _MASK32,
            product1 & _MASK32,
            ((product0 >> 32) ^ counter3 ^ key1) & _MASK32,
            product0 & _MASK32,
        )
        key0 = (key0 + _PHILOX_W0) & _MASK32
        key1 = (key1 + _PHILOX_W1) & _MASK32
    return counter0, counter1, counter2, counter3


def _reference_mask(n, dropout_ratio, seed):
    expected = [[0] * 512 for _ in range(n)]
    threshold = keep_threshold32(dropout_ratio)
    key0 = seed & _MASK32
    key1 = (seed >> 32) & _MASK32
    for row in range(n):
        for column_tile in range(2):
            for lane in range(32):
                philox_block = column_tile * 32 + lane
                for mask_plane in range(3):
                    for half in range(2):
                        words = _philox4x32(row, philox_block * 2 + half, mask_plane, 0, key0, key1)
                        for word_index, word in enumerate(words):
                            column = column_tile * 256 + lane * 8 + half * 4 + word_index
                            if word >= threshold:
                                expected[row][column] |= 1 << mask_plane
    return torch.tensor(expected, dtype=torch.int8)


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


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
@pytest.mark.parametrize(
    "n,dropout_ratio,seed",
    ((1, 0.0, 0), (3, 0.1, 1), (5, 0.37, 2**32 + 17)),
)
def test_forward_mask_matches_philox_reference(n, dropout_ratio, seed):
    torch.manual_seed(29 + n)
    d = 512
    x = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
    u_storage = torch.randn((n, 4 * d), device="cuda", dtype=torch.bfloat16)
    u = u_storage[:, :d]
    weight = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((d,), device="cuda", dtype=torch.bfloat16)

    result = hstu_lmsd_forward(x, u, weight, bias, dropout_ratio=dropout_ratio, seed=seed)
    torch.cuda.synchronize()
    assert torch.equal(result["mask_tensor"].cpu(), _reference_mask(n, dropout_ratio, seed))
