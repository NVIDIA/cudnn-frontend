# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small PyTorch/CPU references for HSTU arbitrary block sparsity tests."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Q2KBlockSparseReference:
    """Exact compact Q-to-K CSR data produced by the reference classifier."""

    mask_block_cnt: torch.Tensor
    mask_block_offset: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: torch.Tensor
    full_block_offset: torch.Tensor
    full_block_idx: torch.Tensor


@dataclass(frozen=True)
class K2QBlockSparseReference:
    """Exact compact K-to-Q CSR data produced by the reference classifier."""

    mask_block_cnt: torch.Tensor
    mask_block_offset: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: torch.Tensor
    full_block_offset: torch.Tensor
    full_block_idx: torch.Tensor


def packed_cu_seqlens(
    lengths: Sequence[int],
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Return int32 packed-sequence offsets for ``lengths``."""

    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + int(length))
    return torch.tensor(offsets, dtype=torch.int32, device=device)


def make_arbitrary_func(
    q_lengths: Sequence[int],
    k_lengths: Sequence[int],
    *,
    pattern: str,
    device: torch.device | str,
) -> torch.Tensor:
    """Build valid HSTU interval endpoints for common test patterns.

    HSTU's odd-sized endpoint vector encodes the union
    ``[0, f0) U [f1, f2) U ...`` for each packed query row.
    """

    if len(q_lengths) != len(k_lengths):
        raise ValueError("q_lengths and k_lengths must have the same length")
    if pattern not in {"empty", "full", "mask", "mixed"}:
        raise ValueError(f"unknown arbitrary-mask pattern: {pattern}")

    func_num = 3 if pattern == "mixed" else 1
    total_q = sum(map(int, q_lengths))
    func = torch.zeros(
        (1, func_num, total_q + 256),
        dtype=torch.int32,
        device=device,
    )

    q_offset = 0
    for q_length, k_length in zip(q_lengths, k_lengths):
        q_length = int(q_length)
        k_length = int(k_length)
        rows = slice(q_offset, q_offset + q_length)
        if pattern == "full":
            func[0, 0, rows] = k_length
        elif pattern == "mask":
            # With the test sequence lengths this partially covers K block 0
            # and leaves every later K block empty.
            func[0, 0, rows] = min(64, k_length)
        elif pattern == "mixed":
            local_q = torch.arange(q_length, device=device)
            row_kind = torch.div(local_q, 64, rounding_mode="floor") % 4

            # Rows of kind 0 are fully enabled.
            func[0, 0, rows] = torch.where(
                row_kind == 0,
                torch.full_like(local_q, k_length, dtype=torch.int32),
                torch.zeros_like(local_q, dtype=torch.int32),
            )
            # Rows of kind 2 enable two partial intervals. Rows of kind 3
            # enable only a suffix. Kind 1 stays empty.
            kind_2 = row_kind == 2
            kind_3 = row_kind == 3
            interval_0_end = min(43, k_length)
            interval_1_begin = min(91, k_length)
            interval_1_end = min(173, k_length)
            suffix_begin = max(k_length - 73, 0)
            func[0, 0, rows] = torch.where(
                kind_2,
                torch.full_like(local_q, interval_0_end, dtype=torch.int32),
                func[0, 0, rows],
            )
            func[0, 1, rows] = torch.where(
                kind_2,
                torch.full_like(local_q, interval_1_begin, dtype=torch.int32),
                torch.where(
                    kind_3,
                    torch.full_like(local_q, suffix_begin, dtype=torch.int32),
                    func[0, 0, rows],
                ),
            )
            func[0, 2, rows] = torch.where(
                kind_2 | kind_3,
                torch.full_like(local_q, interval_1_end, dtype=torch.int32),
                func[0, 0, rows],
            )
        q_offset += q_length
    return func


def arbitrary_dense_masks(
    func: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> list[torch.Tensor]:
    """Expand packed HSTU endpoints into one boolean Q-by-K mask per batch."""

    if func.ndim != 3 or func.shape[0] != 1 or func.shape[1] % 2 != 1:
        raise ValueError("func must have shape (1, positive odd, total_q + padding)")
    q_offsets = [int(x) for x in cu_seqlens_q.detach().cpu().tolist()]
    k_offsets = [int(x) for x in cu_seqlens_k.detach().cpu().tolist()]
    if len(q_offsets) != len(k_offsets):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have matching shapes")

    masks = []
    for batch_idx in range(len(q_offsets) - 1):
        q_begin, q_end = q_offsets[batch_idx : batch_idx + 2]
        k_begin, k_end = k_offsets[batch_idx : batch_idx + 2]
        q_rows = torch.arange(q_begin, q_end, device=func.device)
        k_cols = torch.arange(k_end - k_begin, device=func.device)
        mask = torch.zeros(
            (q_end - q_begin, k_end - k_begin),
            dtype=torch.bool,
            device=func.device,
        )
        for interval in range((int(func.shape[1]) + 1) // 2):
            if interval == 0:
                interval_begin = torch.zeros_like(q_rows, dtype=torch.int32)
            else:
                interval_begin = func[0, 2 * interval - 1, q_rows]
            interval_end = func[0, 2 * interval, q_rows]
            mask |= (k_cols[None, :] >= interval_begin[:, None]) & (k_cols[None, :] < interval_end[:, None])
        masks.append(mask)
    return masks


def q2k_block_sparse_reference(
    func: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size: tuple[int, int],
) -> Q2KBlockSparseReference:
    """Classify EMPTY/MASK/FULL blocks and compact MASK/FULL rows as CSR."""

    tile_m, tile_n = map(int, block_size)
    num_m_blocks = math.ceil(int(max_seqlen_q) / tile_m)
    batch_size = int(cu_seqlens_q.numel() - 1)
    counts_shape = (batch_size, 1, num_m_blocks)
    mask_count = torch.zeros(counts_shape, dtype=torch.int32)
    full_count = torch.zeros(counts_shape, dtype=torch.int32)
    mask_offset = [0]
    full_offset = [0]
    mask_idx: list[int] = []
    full_idx: list[int] = []

    masks = [mask.cpu() for mask in arbitrary_dense_masks(func, cu_seqlens_q, cu_seqlens_k)]
    for batch_idx, mask in enumerate(masks):
        seqlen_q, seqlen_k = mask.shape
        num_n_blocks = math.ceil(seqlen_k / tile_n)
        for m_block in range(num_m_blocks):
            q_begin = m_block * tile_m
            q_end = min(q_begin + tile_m, seqlen_q)
            row_mask_idx: list[int] = []
            row_full_idx: list[int] = []
            if q_begin < q_end:
                for n_block in range(num_n_blocks):
                    k_begin = n_block * tile_n
                    k_end = min(k_begin + tile_n, seqlen_k)
                    block = mask[q_begin:q_end, k_begin:k_end]
                    if bool(block.all()):
                        row_full_idx.append(n_block)
                    elif bool(block.any()):
                        row_mask_idx.append(n_block)

            mask_count[batch_idx, 0, m_block] = len(row_mask_idx)
            full_count[batch_idx, 0, m_block] = len(row_full_idx)
            mask_idx.extend(row_mask_idx)
            full_idx.extend(row_full_idx)
            mask_offset.append(len(mask_idx))
            full_offset.append(len(full_idx))

    return Q2KBlockSparseReference(
        mask_block_cnt=mask_count,
        mask_block_offset=torch.tensor(mask_offset, dtype=torch.int32),
        mask_block_idx=torch.tensor(mask_idx, dtype=torch.int32),
        full_block_cnt=full_count,
        full_block_offset=torch.tensor(full_offset, dtype=torch.int32),
        full_block_idx=torch.tensor(full_idx, dtype=torch.int32),
    )


def k2q_block_sparse_reference(
    func: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size: tuple[int, int] = (128, 128),
) -> K2QBlockSparseReference:
    """Classify logical blocks and compact local Q ids by K work row.

    ``(256, 128)`` models the D256 backward cluster work unit.  Classification
    is performed over the entire logical Q256-by-K128 rectangle, so differing
    Q128 subtile states conservatively produce one MASK supertile.
    """

    tile_q, tile_k = map(int, block_size)
    if tile_q not in (128, 256) or tile_k != 128:
        raise ValueError("K2Q reference supports block_size=(128, 128) or (256, 128)")
    num_k_blocks = math.ceil(int(max_seqlen_k) / tile_k)
    batch_size = int(cu_seqlens_k.numel() - 1)
    counts_shape = (batch_size, 1, num_k_blocks)
    mask_count = torch.zeros(counts_shape, dtype=torch.int32)
    full_count = torch.zeros(counts_shape, dtype=torch.int32)
    mask_offset = [0]
    full_offset = [0]
    mask_idx: list[int] = []
    full_idx: list[int] = []

    masks = [mask.cpu() for mask in arbitrary_dense_masks(func, cu_seqlens_q, cu_seqlens_k)]
    for batch_idx, mask in enumerate(masks):
        seqlen_q, seqlen_k = mask.shape
        num_q_blocks = math.ceil(seqlen_q / tile_q)
        for k_block in range(num_k_blocks):
            k_begin = k_block * tile_k
            k_end = min(k_begin + tile_k, seqlen_k)
            row_mask_idx: list[int] = []
            row_full_idx: list[int] = []
            if k_begin < k_end:
                for q_block in range(num_q_blocks):
                    q_begin = q_block * tile_q
                    q_end = min(q_begin + tile_q, seqlen_q)
                    block = mask[q_begin:q_end, k_begin:k_end]
                    if bool(block.all()):
                        row_full_idx.append(q_block)
                    elif bool(block.any()):
                        row_mask_idx.append(q_block)

            mask_count[batch_idx, 0, k_block] = len(row_mask_idx)
            full_count[batch_idx, 0, k_block] = len(row_full_idx)
            mask_idx.extend(row_mask_idx)
            full_idx.extend(row_full_idx)
            mask_offset.append(len(mask_idx))
            full_offset.append(len(full_idx))

    return K2QBlockSparseReference(
        mask_block_cnt=mask_count,
        mask_block_offset=torch.tensor(mask_offset, dtype=torch.int32),
        mask_block_idx=torch.tensor(mask_idx, dtype=torch.int32),
        full_block_cnt=full_count,
        full_block_offset=torch.tensor(full_offset, dtype=torch.int32),
        full_block_idx=torch.tensor(full_idx, dtype=torch.int32),
    )


def arbitrary_forward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    func: torch.Tensor,
    *,
    alpha: float,
    scaling_seqlen: float,
) -> torch.Tensor:
    """Float32 packed HSTU arbitrary-mask forward reference."""

    q_offsets = [int(x) for x in cu_seqlens_q.detach().cpu().tolist()]
    k_offsets = [int(x) for x in cu_seqlens_k.detach().cpu().tolist()]
    masks = arbitrary_dense_masks(func, cu_seqlens_q, cu_seqlens_k)
    outputs = []
    for batch_idx, mask in enumerate(masks):
        q_begin, q_end = q_offsets[batch_idx : batch_idx + 2]
        k_begin, k_end = k_offsets[batch_idx : batch_idx + 2]
        q_batch = q[q_begin:q_end].float()
        k_batch = k[k_begin:k_end].float()
        v_batch = v[k_begin:k_end].float()
        scores = float(alpha) * torch.einsum(
            "qhd,khd->hqk",
            q_batch,
            k_batch,
        )
        weights = torch.where(
            mask.unsqueeze(0),
            F.silu(scores),
            torch.zeros_like(scores),
        )
        outputs.append(torch.einsum("hqk,khd->qhd", weights, v_batch) / float(scaling_seqlen))
    return torch.cat(outputs, dim=0)
