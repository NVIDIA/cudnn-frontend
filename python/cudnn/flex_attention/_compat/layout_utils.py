# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# SPDX-License-Identifier: Apache-2.0
"""Minimal layout helpers adapted from quack-kernels 0.5.0."""

import cutlass.cute as cute
from cutlass import const_expr


def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Transpose the first two dimensions of a tensor on smem."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


def select(a: cute.Tensor, mode: list[int]) -> cute.Tensor:
    return cute.make_tensor(a.iterator, cute.select(a.layout, mode))


def convert_layout_acc_mn(acc_layout: cute.Layout, transpose: bool = False) -> cute.Layout:
    """Convert an SM90 accumulator layout into an M/N-major view."""
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    shape = (
        (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),  # MMA_M
        (
            acc_layout_col_major.shape[0][0],
            *acc_layout_col_major.shape[0][2:],
            acc_layout_col_major.shape[2],
        ),  # MMA_N
        *acc_layout_col_major.shape[3:],
    )
    stride = (
        (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),  # MMA_M
        (
            acc_layout_col_major.stride[0][0],
            *acc_layout_col_major.stride[0][2:],
            acc_layout_col_major.stride[2],
        ),  # MMA_N
        *acc_layout_col_major.stride[3:],
    )
    if const_expr(transpose):
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    acc_layout_mn = cute.make_layout(shape, stride=stride)
    return cute.composition(acc_layout, acc_layout_mn)


def reshape_acc_to_mn(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout, transpose=transpose))


@cute.jit
def convert_layout_acc_frgA(acc_layout: cute.Layout) -> cute.Layout:
    # For back-to-back GEMMs, convert the first accumulator to the fragment-A layout.
    # For SM90 FP16/BF16, convert ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2, 2), MMA_M, (N / 16, MMA_N)).
    # If N / 8 is odd, we'll convert to ((2, 2, 1), MMA_M, N / 8, MMA_N).
    assert cute.rank(acc_layout.shape[0]) == 3
    div = 2 if const_expr(acc_layout.shape[0][2] % 2 == 0) else 1
    l = cute.logical_divide(acc_layout, ((None, None, div), None, None))  # ((2, 2, (2, N / 16)), MMA_M, MMA_N)
    return cute.make_layout(
        (
            (l.shape[0][0], l.shape[0][1], l.shape[0][2][0]),
            l.shape[1],
            (l.shape[0][2][1], l.shape[2]),
        ),
        stride=(
            (l.stride[0][0], l.stride[0][1], l.stride[0][2][0]),
            l.stride[1],
            (l.stride[0][2][1], l.stride[2]),
        ),
    )


def reshape_acc_to_frgA(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_frgA(acc.layout))


def mma_partition_C_vec(sVec: cute.Tensor, thr_mma: cute.ThrMma, expand_shape: int, is_colvec: bool) -> cute.Tensor:
    assert cute.rank(sVec) == 2
    assert sVec.stride[0] == 1
    stage = sVec.shape[1]
    shape = (sVec.shape[0], expand_shape, stage) if const_expr(is_colvec) else (expand_shape, sVec.shape[0], stage)
    stride = (1, 0, sVec.stride[1]) if const_expr(is_colvec) else (0, 1, sVec.stride[1])
    sVec_mma = cute.make_tensor(sVec.iterator, cute.make_layout(shape, stride=stride))
    tC_sVec = reshape_acc_to_mn(thr_mma.partition_C(sVec_mma))
    return tC_sVec[None, 0, None] if const_expr(is_colvec) else tC_sVec[0, None, None]
