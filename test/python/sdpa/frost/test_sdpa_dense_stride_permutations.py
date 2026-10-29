# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native dense descriptor coordinates preserve each physical B/H/S order."""

from itertools import permutations, product

import pytest

from cudnn.sdpa.fwd.config_sm100 import dense_bind_strides
from cudnn.sdpa.fwd.prepared import _dense_role_layout

pytestmark = [pytest.mark.L0]


@pytest.mark.parametrize("order", list(permutations(range(3))))
@pytest.mark.parametrize("gap", [0, 8, 24])
def test_native_dense_stride_permutations_preserve_addresses(order, gap):
    """Check the native binder against the actual declared element-address set."""
    shape = (2, 3, 5, 16)
    strides = [0, 0, 0, 1]
    span = shape[-1] + gap
    for axis in reversed(order):
        strides[axis] = span
        span = shape[axis] * span + gap
    bound, batch, sequence, need = _dense_role_layout(shape, tuple(strides), 3, 16, 5, 2, 2, True, "q")
    bs, ss, hs = bound
    assert (batch, sequence) == (2, 5)
    declared = {sum(i * st for i, st in zip(index, strides, strict=True)) for index in product(*(range(n) for n in shape))}
    addressed = {b * bs + h * hs + s * ss + d for b, h, s, d in product(*(range(n) for n in shape))}
    assert addressed == declared
    assert len(addressed) == 2 * 3 * 5 * 16
    assert need == max(addressed) + 1


@pytest.mark.parametrize("unused", [0, 3, -7, 2**33 + 1])
def test_native_dense_singleton_batch_does_not_constrain_physical_order(unused):
    assert dense_bind_strides((1, 3, 5, 16), (unused, 80, 16, 1), 2) == (80, 16, 80)


@pytest.mark.parametrize("strides", [(256, 80, 17, 1), (256, 79, 16, 1), (255, 80, 16, 1), (256, 80, 16, 2), (256, 64, 16, 1)])
def test_native_dense_declines_unaligned_or_overlapping_axes(strides):
    assert dense_bind_strides((2, 3, 5, 16), strides, 2) is None
