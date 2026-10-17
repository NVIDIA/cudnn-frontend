# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side bounds of the scratch shared by steep-decay producers and consumers."""

import dataclasses

import pytest

from cudnn.linear_attention.frost.common.steep_scratch import steep_scratch_layout, state_gradient_scratch_layout

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("d_k", [64, 128])
@pytest.mark.parametrize("stages", [1, 2, 3])
@pytest.mark.parametrize("gradients", [False, True])
def test_steep_scratch_regions(d_k, stages, gradients):
    layout = steep_scratch_layout(
        b_t=16,
        d_k=d_k,
        scan_warps=4,
        scan_stages=stages,
        last_stages=stages,
        decay_stages=stages,
        norm_stages=0 if gradients else 1,
        dadm_stages=stages if gradients else 0,
    )
    occupied = set()
    for field in dataclasses.fields(layout):
        if field.name == "nbytes":
            continue
        region = getattr(layout, field.name)
        assert region.offset % 16 == 0
        assert region.offset + region.nbytes <= layout.nbytes
        addresses = set(range(region.offset, region.offset + region.nbytes))
        assert not occupied.intersection(addresses), field.name
        occupied.update(addresses)
        assert region.nbytes == region.elements * (2 if field.name == "dadm" else 4)
    assert layout.last.elements == stages * d_k
    assert layout.dadm.elements == (stages * 512 if gradients else 0)
    assert layout.nbytes % 16 == 0
    layout.check_capacity(layout.nbytes)
    with pytest.raises(ValueError, match="steep scratch needs"):
        layout.check_capacity(layout.nbytes - 1)


@pytest.mark.parametrize("d_k", [64, 128])
@pytest.mark.parametrize("d_v", [64, 128])
@pytest.mark.parametrize("stages", [1, 2, 3])
def test_state_gradient_scratch_is_disjoint(d_k, d_v, stages):
    layout = state_gradient_scratch_layout(d_k=d_k, d_v=d_v, b_t=16, decay_stages=stages)
    occupied = set()
    # Model the actual producer indices across adjacent persistent CTAs.
    for cta in range(2):
        base = cta * layout.nbytes
        for offset, rows in [(layout.state.offset, d_v), (layout.keys.offset, stages * 16)]:
            assert offset % 16 == 0
            addresses = {base + offset + (row * d_k + channel) * 4 for row in range(rows) for channel in range(d_k)}
            assert not occupied.intersection(addresses)
            assert max(addresses) + 4 <= base + layout.nbytes
            occupied.update(addresses)
