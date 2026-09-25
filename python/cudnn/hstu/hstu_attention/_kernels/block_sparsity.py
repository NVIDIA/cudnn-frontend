# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private block-sparse metadata used by the native HSTU kernels.

The public HSTU API deliberately does not expose these objects.  They are built
from ``func_tensor`` by ``_interface.py`` and live only for the duration of one
attention call.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Boolean, Int32


class HSTUBlockSparseTensors(NamedTuple):
    """CuTe view of one compact CSR orientation."""

    mask_block_cnt: cute.Tensor
    mask_block_offset: cute.Tensor
    mask_block_idx: cute.Tensor
    full_block_cnt: cute.Tensor
    full_block_offset: cute.Tensor
    full_block_idx: cute.Tensor

    def __new_from_mlir_values__(self, values):
        return HSTUBlockSparseTensors(*values)


class HSTUBlockSparseTensorsTorch(NamedTuple):
    """Torch-owned storage for one compact CSR orientation."""

    mask_block_cnt: torch.Tensor
    mask_block_offset: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: torch.Tensor
    full_block_offset: torch.Tensor
    full_block_idx: torch.Tensor
    block_size: tuple[int, int]
    orientation: str


class HSTUBlockSparseBuilderWorkspace(NamedTuple):
    """Capacity workspace used while compacting a CSR orientation."""

    tensors: HSTUBlockSparseTensorsTorch
    mask_staging: torch.Tensor
    full_staging: torch.Tensor
    mask_scan_blocks: torch.Tensor
    full_scan_blocks: torch.Tensor


class HSTUK2QBlockSparseBuilderWorkspace(NamedTuple):
    """Capacity workspace used while building transposed K-to-Q CSR."""

    tensors: HSTUBlockSparseTensorsTorch
    mask_staging: torch.Tensor
    full_staging: torch.Tensor
    mask_scan_blocks: torch.Tensor
    full_scan_blocks: torch.Tensor
    block_states: torch.Tensor


class HSTUD256BwdBlockSparseBuilderWorkspace(NamedTuple):
    """Workspace for one shared Q256-by-K128 backward classification.

    ``k2q.block_states`` owns the shared coarse state matrix.  The paired
    builder classifies it once, then compacts the matrix in both orientations
    into the independent Q2K and K2Q capacity-backed CSR tensors.
    """

    q2k: HSTUBlockSparseBuilderWorkspace
    k2q: HSTUK2QBlockSparseBuilderWorkspace


@cute.jit
def get_q2k_block_sparse_row(
    tensors: HSTUBlockSparseTensors,
    batch_idx: Int32,
    m_block: Int32,
) -> Tuple[Int32, cute.Tensor, Int32, cute.Tensor]:
    """Return the mask/full lists for one batch-local Q work row."""

    num_m_blocks = tensors.mask_block_cnt.shape[2]
    flat_row = batch_idx * num_m_blocks + m_block
    mask_count = tensors.mask_block_cnt[batch_idx, 0, m_block]
    full_count = tensors.full_block_cnt[batch_idx, 0, m_block]
    mask_indices = cute.domain_offset(
        tensors.mask_block_offset[flat_row],
        tensors.mask_block_idx,
    )
    full_indices = cute.domain_offset(
        tensors.full_block_offset[flat_row],
        tensors.full_block_idx,
    )
    return mask_count, mask_indices, full_count, full_indices


@cute.jit
def get_q2k_block_for_iteration(
    mask_count: Int32,
    mask_indices: cute.Tensor,
    full_count: Int32,
    full_indices: cute.Tensor,
    iteration: Int32,
) -> Tuple[Int32, Boolean]:
    """Map a mask-then-full traversal position to a batch-local K block.

    Each list is consumed in reverse block order to preserve the native HSTU
    pipeline's existing high-to-low K traversal.
    """

    total_count = mask_count + full_count
    is_mask = Boolean(iteration < mask_count)
    n_block = Int32(0)
    if iteration < mask_count:
        n_block = mask_indices[mask_count - 1 - iteration]
    elif iteration < total_count:
        full_iteration = iteration - mask_count
        n_block = full_indices[full_count - 1 - full_iteration]
    return n_block, is_mask


@cute.jit
def get_q2k_block_sparse_consumer_row(
    tensors: HSTUBlockSparseTensors,
    batch_idx: Int32,
    m_block: Int32,
) -> Tuple[Int32, Int32, cute.Tensor, Int32, cute.Tensor]:
    """Return one row plus the iteration count shared by all warp roles."""

    mask_count, mask_indices, full_count, full_indices = get_q2k_block_sparse_row(tensors, batch_idx, m_block)
    # Keep the exact count.  Consumers branch on zero before forming count - 1
    # or touching either CSR index list.
    consumer_count = mask_count + full_count
    return (
        consumer_count,
        mask_count,
        mask_indices,
        full_count,
        full_indices,
    )


@cute.jit
def get_q2k_block_for_reverse_slot(
    consumer_count: Int32,
    mask_count: Int32,
    mask_indices: cute.Tensor,
    full_count: Int32,
    full_indices: cute.Tensor,
    reverse_slot: Int32,
) -> Tuple[Int32, Boolean]:
    """Map the native high-to-low loop slot to mask-then-full traversal."""

    iteration = consumer_count - 1 - reverse_slot
    return get_q2k_block_for_iteration(
        mask_count,
        mask_indices,
        full_count,
        full_indices,
        iteration,
    )


@cute.jit
def get_k2q_block_sparse_consumer_row(
    tensors: HSTUBlockSparseTensors,
    batch_idx: Int32,
    k_cluster: Int32,
) -> Tuple[Int32, Int32, cute.Tensor, Int32, cute.Tensor]:
    """Return one K2Q cluster row plus its exact coarse-tile count.

    K2Q tensors use the same capacity-backed CSR representation as Q2K, but
    their row coordinate is a batch-local K work tile.  In the D256 backward
    dK/dV kernel this coordinate is the shared K128 cluster id, not the
    physical K64 CTA id.
    """

    num_k_clusters = tensors.mask_block_cnt.shape[2]
    flat_row = batch_idx * num_k_clusters + k_cluster
    mask_count = tensors.mask_block_cnt[batch_idx, 0, k_cluster]
    full_count = tensors.full_block_cnt[batch_idx, 0, k_cluster]
    mask_indices = cute.domain_offset(
        tensors.mask_block_offset[flat_row],
        tensors.mask_block_idx,
    )
    full_indices = cute.domain_offset(
        tensors.full_block_offset[flat_row],
        tensors.full_block_idx,
    )
    return (
        mask_count + full_count,
        mask_count,
        mask_indices,
        full_count,
        full_indices,
    )


@cute.jit
def get_k2q_q_block_for_subtile_iteration(
    mask_count: Int32,
    mask_indices: cute.Tensor,
    full_count: Int32,
    full_indices: cute.Tensor,
    iteration: Int32,
    subtile_factor: cutlass.Constexpr[int],
) -> Tuple[Int32, Boolean]:
    """Expand one K2Q coarse Q tile into a physical Q-subtile iteration.

    Lists are consumed MASK then FULL and in their builder-defined ascending
    order.  For D256 dK/dV, ``subtile_factor`` is two: a Q256 metadata index
    expands to two consecutive Q128 compute iterations.  Every warp role and
    both paired K64 CTAs call this helper with the same cluster row and
    iteration, which keeps their pipeline phases in lockstep.
    """

    coarse_iteration = iteration // subtile_factor
    subtile = iteration % subtile_factor
    coarse_q_block = Int32(0)
    is_mask = Boolean(coarse_iteration < mask_count)
    if coarse_iteration < mask_count:
        coarse_q_block = mask_indices[coarse_iteration]
    elif coarse_iteration < mask_count + full_count:
        coarse_q_block = full_indices[coarse_iteration - mask_count]
    return coarse_q_block * subtile_factor + subtile, is_mask
