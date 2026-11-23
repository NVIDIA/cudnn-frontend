# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# SPDX-License-Identifier: Apache-2.0
"""Minimal layout helpers adapted from quack-kernels 0.5.0."""

import cutlass.cute as cute
from cutlass import const_expr


def tile_atom_to_shape_SF_strided(
    shape: cute.Shape,
    sf_vec_size: int,
    sf_strides,
) -> cute.Layout:
    """Build an SFA/SFB layout matching `shape` (A or B operand shape) but
    honoring the scale tensor's actual strides instead of hardcoded packed
    ones.

    Mirrors `cutlass.utils.blockscaled_layout.tile_atom_to_shape_SF(shape,
    sf_vec_size)`, except outer-mode strides come from `sf_strides` (pass
    `mSFA.stride` / `mSFB.stride` directly). The inner 512-B atom
    `((32, 4), (sf_vec_size, 4)) : ((16, 4), (0, 1))` is hardware-fixed.

    Implementation uses `cute.blocked_product(atom, outer)`; `blocked_product`
    scales the outer layout's strides by `cosize(atom) == 512`, so we divide
    the byte strides by 512 (one tile) before handing them in.

    Args:
        shape: A/B operand shape. Rank-3 `(m/n, k, l)` or rank-2
            `(total_mn, k)` (varlen_m).
        sf_vec_size: Scale factor vector size (16 or 32).
        sf_strides: Strides of the scale tensor, which has logical shape
            `(L, rmn, rk, 512)` (rank 4). Only `sf_strides[0..2]` are used:
            `sf_strides[1]` as the rmn stride, `sf_strides[2]` as the rk
            stride, and `sf_strides[0]` as the L stride (only for rank-3
            `shape`).
    """
    from cutlass.utils.blockscaled_layout import BlockScaledBasicChunk

    atom = BlockScaledBasicChunk(sf_vec_size).layout
    rmn = cute.ceil_div(shape[0], 128)
    rk = cute.ceil_div(shape[1], sf_vec_size * 4)
    outer = cute.make_layout((rmn, rk), stride=(sf_strides[1] // 512, sf_strides[2] // 512))
    sf_layout = cute.blocked_product(atom, outer)
    if const_expr(len(shape) == 3):
        sf_layout = cute.append(sf_layout, cute.make_layout(shape[2], stride=sf_strides[0]))
    return sf_layout
