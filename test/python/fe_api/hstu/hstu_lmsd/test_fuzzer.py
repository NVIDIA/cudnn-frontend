# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shape and dropout fuzz coverage for HSTU LMSD forward."""

import pytest
import torch

try:
    import cutlass  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(f"CuTe DSL is unavailable: {exc}", allow_module_level=True)

from cudnn.hstu.hstu_lmsd import hstu_lmsd_forward
from cudnn.hstu.hstu_lmsd._kernels._common import keep_threshold32
from reference import hstu_lmsd_forward_reference

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


def _reference_mask(n, d, dropout_ratio, seed):
    expected = [[0] * d for _ in range(n)]
    threshold = keep_threshold32(dropout_ratio)
    key0 = seed & _MASK32
    key1 = (seed >> 32) & _MASK32
    threads_per_row = 32
    vector_size = 8
    philox_groups = vector_size // 4
    num_vectors = d // vector_size
    num_column_tiles = (num_vectors + threads_per_row - 1) // threads_per_row
    for row in range(n):
        for column_tile in range(num_column_tiles):
            for lane in range(threads_per_row):
                philox_block = column_tile * threads_per_row + lane
                if philox_block >= num_vectors:
                    continue
                for mask_plane in range(3):
                    for group in range(philox_groups):
                        words = _philox4x32(row, philox_block * philox_groups + group, mask_plane, 0, key0, key1)
                        for word_index, word in enumerate(words):
                            column = column_tile * threads_per_row * vector_size + lane * vector_size + group * 4 + word_index
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
    mask_i32 = mask.to(torch.int32) if mask is not None else None
    if mask_i32 is not None:
        assert torch.count_nonzero(mask_i32 & ~0x7) == 0

    expected_y, expected_mean, expected_rstd = hstu_lmsd_forward_reference(x, u, weight, bias, mask, dropout_ratio, eps)
    torch.testing.assert_close(mean, expected_mean, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(rstd, expected_rstd, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(y.float(), expected_y, rtol=1.5e-2, atol=1.5e-2)


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
@pytest.mark.parametrize(
    "d,n,dropout_ratio,seed",
    ((8, 1, 0.0, 0), (24, 3, 0.1, 1), (264, 5, 0.37, 2**32 + 17), (1016, 2, 0.1, 23)),
)
def test_forward_mask_matches_philox_reference(d, n, dropout_ratio, seed):
    torch.manual_seed(29 + n)
    x = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
    u_storage = torch.randn((n, 4 * d), device="cuda", dtype=torch.bfloat16)
    u = u_storage[:, :d]
    weight = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((d,), device="cuda", dtype=torch.bfloat16)

    result = hstu_lmsd_forward(x, u, weight, bias, dropout_ratio=dropout_ratio, seed=seed)
    torch.cuda.synchronize()
    if dropout_ratio == 0.0:
        assert result["mask_tensor"] is None
    else:
        assert torch.equal(result["mask_tensor"].cpu(), _reference_mask(n, d, dropout_ratio, seed))
