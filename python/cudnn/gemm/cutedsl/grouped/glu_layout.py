# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLU branch layout contracts and zero-copy CuTe views.

These views change addresses, not storage. ``gate_then_up`` is the persistent
parameter/checkpoint layout; ``interleaved_32`` is the existing MMA/epilogue order.
"""

import cutlass
import cutlass.cute as cute

GLU_LAYOUTS = ("interleaved_32", "gate_then_up")


def validate_glu_layout(glu_layout: str) -> str:
    if glu_layout not in GLU_LAYOUTS:
        raise ValueError(f"glu_layout must be one of {GLU_LAYOUTS}, got {glu_layout!r}")
    return glu_layout


@cute.jit
def interleaved_glu_view(tensor: cute.Tensor, axis: cutlass.Constexpr, block_size: cutlass.Constexpr = 32):
    """View a flat canonical branch axis in paired ``block_size`` chunks.

    Logical ``r + block_size*p + 2*block_size*q`` addresses canonical
    ``r + (N/2)*p + block_size*q``. The caller validates N divisibility.
    """
    extent = tensor.shape[axis]
    stride = tensor.stride[axis]
    shape = tuple((block_size, 2, extent // (2 * block_size)) if i == axis else tensor.shape[i] for i in range(cute.rank(tensor)))
    strides = tuple((stride, (extent // 2) * stride, block_size * stride) if i == axis else tensor.stride[i] for i in range(cute.rank(tensor)))
    return cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=strides))


@cute.jit
def paired_glu_sfb_view(tensor: cute.Tensor):
    """Pair whole canonical F8_128x4 scale atoms for one 256-column tile.

    TMA cannot gather the 4-byte quarters needed by interleave32. Load one
    gate128/up128 pair and let the MMA warp rearrange those quarters in SMEM.
    This view preserves all 16-byte contiguous pieces required by TMA.
    """
    rest_n = tensor.shape[0][1]
    rest_stride = tensor.stride[0][1]
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(
            ((tensor.shape[0][0], (2, rest_n // 2)), tensor.shape[1], tensor.shape[2]),
            stride=((tensor.stride[0][0], ((rest_n // 2) * rest_stride, rest_stride)), tensor.stride[1], tensor.stride[2]),
        ),
    )


@cute.jit
def interleave_sfb_in_place(
    sSFB: cute.Tensor, stage: cutlass.Int32, lane: cutlass.Int32, tile_k: cutlass.Constexpr, sf_vec_size: cutlass.Constexpr, num_stages: cutlass.Constexpr
):
    """Pair scale quarters after TMA, before the existing SMEM->TMEM copy.

    Each lane owns one of the 32 rows in both 128-row atoms and loads all
    eight 4-byte quarters before overwriting any. The transform is entirely
    within a lane. ``tile_atom_to_shape_SF`` orders K atoms before N atoms.
    """
    k_atoms = tile_k // (4 * sf_vec_size)
    words = cute.make_tensor(
        cute.recast_ptr(sSFB.iterator, dtype=cutlass.Uint32),
        cute.make_layout(
            (32, 4, 2, k_atoms, num_stages),
            stride=(4, 1, 128 * k_atoms, 128, 256 * k_atoms),
        ),
    )
    values = cute.make_rmem_tensor((8,), cutlass.Uint32)
    for k_atom in cutlass.range_constexpr(k_atoms):
        for branch in cutlass.range_constexpr(2):
            for quarter in cutlass.range_constexpr(4):
                values[branch * 4 + quarter] = words[(lane, quarter, branch, k_atom, stage)]
        for quarter in cutlass.range_constexpr(4):
            for branch in cutlass.range_constexpr(2):
                paired = 2 * quarter + branch
                words[(lane, paired % 4, paired // 4, k_atom, stage)] = values[branch * 4 + quarter]
    cute.arch.sync_warp()
    cute.arch.fence_proxy("async.shared", space="cta")
