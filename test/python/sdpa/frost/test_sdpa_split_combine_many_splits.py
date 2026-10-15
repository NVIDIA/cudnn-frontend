# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The shared split combine over MANY splits: the grouped walks (the lean forms' up-front loads of at most ``G_S`` splits, the serial and
decode forms' walks past them) reduce every split count -- a group boundary hit exactly, one short of it, one past it, and a decode-shaped
row count -- to the float64 reference, with dead splits and dead rows handled as the three-split tests of ``test_sdpa_split_combine_sm100.py``
pin them -- the decode-shaped case keeps its only row LIVE (a dead split inside the walk and a dead LAST split beside live ones), so the
grouped accumulation on a live decode row is what it checks; the multi-row shape adds the all-dead row.  Runs on any
SM100-line part (the combine has no arch-specific lowering; the Rubin d256 decode tile is its heaviest caller at 64 splits)."""

import pytest

import torch

from cudnn.frost.compiled_cache import positional_entry

from frost_test_utils import requires_blackwell, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell]


def _partials(b, h, sq, d, splits):
    generator = torch.Generator().manual_seed(20261014 + splits)
    o = torch.randn(splits, b, sq, h, d, generator=generator) * 0.4
    lse = torch.randn(splits, b, h, sq, generator=generator) * 2.0 + 0.7
    if sq > 1:
        lse[:, :, :, 0] = -torch.inf  # every split dead on one row (a multi-row shape); the decode shape keeps its ONLY row live
    lse[1, :, :, min(1, sq - 1)] = -torch.inf  # a dead split beside live ones
    lse[splits - 1, :, 0, :] = -torch.inf  # the LAST split dead on one head: the clamped tail slots re-read it
    dead = torch.isneginf(lse).permute(0, 1, 3, 2).unsqueeze(-1)
    o.masked_fill_(dead, torch.nan)  # a dead payload must never enter the arithmetic
    scores = lse.double()
    maximum = scores.amax(0)
    safe_maximum = torch.where(torch.isfinite(maximum), maximum, 0)
    weights = (scores - safe_maximum).exp()
    denominator = weights.sum(0)
    safe_o = o.double().masked_fill(dead, 0)
    ref_o = (weights.permute(0, 1, 3, 2).unsqueeze(-1) * safe_o).sum(0) / denominator.clamp_min(1).permute(0, 2, 1).unsqueeze(-1)
    ref_lse = safe_maximum + denominator.log()
    return o.reshape(splits * b, sq, h, d).cuda(), lse.reshape(splits * b, h, sq).cuda(), ref_o, ref_lse


@pytest.mark.parametrize("splits", [4, 5, 16, 17, 20, 33, 64, 65], ids=lambda s: f"s{s}")
@pytest.mark.parametrize("shape", [(1, 24, 1, 256), (2, 3, 5, 160)], ids=["decode_b1_h24_d256", "b2_h3_sq5_d160"])
def test_pointer_combine_many_splits_matches_the_reference(shape, splits):
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d = shape
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    o = torch.full((b, sq, h, d), -31.0, device="cuda", dtype=torch.bfloat16)
    lse = torch.full((b, h, sq), -31.0, device="cuda", dtype=torch.float32)
    fn = positional_entry(comb.compile_ptr(dtype_o="bf16", has_lse=True))
    assert fn is not None
    fn(op.data_ptr(), lp.data_ptr(), o.data_ptr(), lse.data_ptr(), (b, h, sq, d), splits, o.stride(), lse.stride(), torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    torch.testing.assert_close(o.cpu().float(), ref_o.float(), atol=0.004, rtol=0.004)
    if sq > 1:
        assert torch.count_nonzero(o[:, 0]).item() == 0  # the all-dead row is exactly 0
    torch.testing.assert_close(lse.cpu().double(), ref_lse, atol=2e-5, rtol=2e-5)
    if sq > 1:
        assert torch.all(torch.isneginf(lse[:, :, 0]))


def test_the_walks_load_several_splits_per_round_trip():
    """The grouped walk is the lever (a one-split-per-step walk cost ~0.3 us per split per pass on a batch-1 decode launch): the lean forms
    load every split of a row up front (``G_LEAN_SMALL`` / ``G_S`` of them), the serial form walks ``_SERIAL_UNROLL`` deep and the decode
    form's O pass ``_DECODE_UNROLL`` deep."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    assert comb.G_LEAN_SMALL >= 8 and comb.G_S >= comb.G_LEAN_SMALL
    assert comb._SERIAL_UNROLL >= 8 and comb._DECODE_UNROLL >= 8
