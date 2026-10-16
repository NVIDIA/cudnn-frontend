# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit, standalone blk64 -> blk128 grouping (not a FROST graph engine).

Preparation snapshots and analyzes the sparse graph on the CPU. The resulting
metadata belongs to the caller and must be rebuilt when the graph changes.
Torch wrappers allocate their outputs/scratch, just like the ordinary BSA API.
After warmup they can be captured; replay includes all pack/unpack kernels.
"""

from collections import defaultdict
from dataclasses import dataclass
import math

import torch

from cudnn.api_base import TupleDict


@dataclass(frozen=True)
class _GroupedMetadata:
    # Packed block -> original block, independently for every batch and head.
    q_permutation: torch.Tensor
    kv_permutation: torch.Tensor
    q2k_block_index: torch.Tensor
    q2k_block_nums: torch.Tensor
    bucketed_k2q_offsets: torch.Tensor
    bucketed_k2q_indices: torch.Tensor
    fixed_count: int | None
    identity_q: bool
    identity_kv: bool
    bucket_size_blocks: int
    edges64: int


def _pair_groups(groups, axis):
    permutation = []
    for members in groups:
        if len(members) % 2:
            raise ValueError(f"fully paired grouping requires every {axis} neighborhood to occur an even number of times")
        permutation.extend(members)
    return permutation


def _coarsen_head(rows, counts, num_kv_blocks):
    """Exact equality after sorting valid prefixes; hashes never establish equality."""
    q_groups = defaultdict(list)
    for row, (indices, count) in enumerate(zip(rows, counts)):
        if not 0 < count <= len(indices):
            raise ValueError("Q rows must be nonempty and counts must not exceed the route capacity")
        valid = tuple(sorted(indices[:count]))
        if valid[0] < 0 or valid[-1] >= num_kv_blocks:
            raise ValueError("valid KV indices must be in [0, num_kv_blocks)")
        if any(a == b for a, b in zip(valid, valid[1:])):
            raise ValueError("duplicate KV indices in a valid row are unsupported")
        q_groups[valid].append(row)
    q_perm = _pair_groups(q_groups.values(), "Q")

    # Reverse neighborhoods can be represented by unique Q-neighborhood IDs:
    # all Q blocks within one group have exactly the same outgoing edges.
    reverse = [[] for _ in range(num_kv_blocks)]
    for group_id, indices in enumerate(q_groups):
        for kv in indices:
            reverse[kv].append(group_id)
    kv_groups = defaultdict(list)
    for kv, neighbors in enumerate(reverse):
        kv_groups[tuple(neighbors)].append(kv)
    kv_perm = _pair_groups(kv_groups.values(), "KV")
    kv_pair = [0] * num_kv_blocks
    for packed, original in enumerate(kv_perm):
        kv_pair[original] = packed // 2

    coarse = []
    for indices, members in q_groups.items():
        paired_indices = sorted({kv_pair[kv] for kv in indices})
        if 2 * len(paired_indices) != len(indices):
            raise ValueError("coarsening would introduce a missing KV64 edge")
        coarse.extend([paired_indices] * (len(members) // 2))
    return q_perm, kv_perm, coarse


def prepare(q2k_block_index, block_sparse_num, *, num_kv_blocks, q2k_block_nums, bucket_size_blocks):
    if not isinstance(q2k_block_index, torch.Tensor) or q2k_block_index.ndim != 4 or q2k_block_index.dtype != torch.int32:
        raise ValueError("q2k_block_index must be a rank-4 CUDA int32 tensor")
    if not q2k_block_index.is_cuda:
        raise ValueError("q2k_block_index must be a CUDA tensor")
    b, h, nq, capacity = q2k_block_index.shape
    if min(b, h, nq, capacity) < 1 or nq % 2:
        raise ValueError("positive dimensions and an even number of full Q64 blocks are required")
    if not isinstance(num_kv_blocks, int) or isinstance(num_kv_blocks, bool) or num_kv_blocks <= 0 or num_kv_blocks % 2:
        raise ValueError("num_kv_blocks must be a positive even number of full KV64 blocks")
    if nq * capacity >= 2**31 or num_kv_blocks >= 2**31:
        raise ValueError("sparse metadata exceeds the int32 CSR indexing range")
    if max(nq, num_kv_blocks) * 64 >= 2**31:
        raise ValueError("sequence lengths exceed the int32 token indexing range")
    if q2k_block_nums is None:
        block_sparse_num = capacity if block_sparse_num is None else block_sparse_num
        if not isinstance(block_sparse_num, int) or isinstance(block_sparse_num, bool) or not 0 < block_sparse_num <= capacity:
            raise ValueError("block_sparse_num must be a positive integer no larger than capacity")
    elif (
        not isinstance(q2k_block_nums, torch.Tensor)
        or q2k_block_nums.dtype != torch.int32
        or tuple(q2k_block_nums.shape) != (b, h, nq)
        or q2k_block_nums.device != q2k_block_index.device
    ):
        raise ValueError("q2k_block_nums must be int32 (B, H, Q64 blocks) on the same CUDA device as the indices")
    if bucket_size_blocks is None:
        # Match the native blk128 default without importing any DSL kernels.
        bucket_size_blocks = 256 if nq // 2 >= 4096 and h <= 1 else 512 if nq // 2 >= 2048 else 384
    if not isinstance(bucket_size_blocks, int) or isinstance(bucket_size_blocks, bool) or bucket_size_blocks <= 0:
        raise ValueError("bucket_size_blocks must be a positive integer in Q128 units")

    device = q2k_block_index.device
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("grouped_prepare performs CPU graph analysis; call it before CUDA Graph capture")
        # Explicit blocking preprocessing, never called from forward/backward.
        indices_cpu = q2k_block_index.detach().cpu()
        counts_cpu = q2k_block_nums.detach().cpu() if q2k_block_nums is not None else torch.full((b, h, nq), block_sparse_num, dtype=torch.int32)
        q_perms, kv_perms, coarse_heads = [], [], []
        for batch in range(b):
            for head in range(h):
                qp, kp, coarse = _coarsen_head(indices_cpu[batch, head].tolist(), counts_cpu[batch, head].tolist(), num_kv_blocks)
                q_perms.append(qp)
                kv_perms.append(kp)
                coarse_heads.append(coarse)

        coarse_capacity = max(len(row) for head in coarse_heads for row in head)
        counts = torch.tensor([[len(row) for row in head] for head in coarse_heads], dtype=torch.int32).reshape(b, h, nq // 2)
        routes = torch.tensor([[row + [0] * (coarse_capacity - len(row)) for row in head] for head in coarse_heads], dtype=torch.int32).reshape(
            b, h, nq // 2, coarse_capacity
        )
        groups = (nq // 2 + bucket_size_blocks - 1) // bucket_size_blocks
        all_offsets, all_edges = [], []
        for head in coarse_heads:
            offsets, edges = [], []
            for group in range(groups):
                owners = [[] for _ in range(num_kv_blocks // 2)]
                start = group * bucket_size_blocks
                for q in range(start, min(start + bucket_size_blocks, nq // 2)):
                    for kv in head[q]:
                        owners[kv].append(q)
                group_offsets = [len(edges)]
                for neighbors in owners:
                    edges.extend(neighbors)
                    group_offsets.append(len(edges))
                offsets.append(group_offsets)
            all_offsets.append(offsets)
            all_edges.append(edges)
        edge_capacity = max(map(len, all_edges))
        offsets = torch.tensor(all_offsets, dtype=torch.int32).reshape(b, h, groups, num_kv_blocks // 2 + 1)
        edges = torch.tensor([row + [0] * (edge_capacity - len(row)) for row in all_edges], dtype=torch.int32).reshape(b, h, edge_capacity)
        fixed_count = coarse_capacity if coarse_capacity % 2 == 0 and bool((counts == coarse_capacity).all()) else None
        edges64 = int(counts_cpu.sum())
        if edges64 != 4 * int(counts.sum()):
            raise ValueError("coarsening did not preserve the edge count")
        return _GroupedMetadata(
            torch.tensor(q_perms, dtype=torch.int32, device=device).reshape(b, h, nq),
            torch.tensor(kv_perms, dtype=torch.int32, device=device).reshape(b, h, num_kv_blocks),
            routes.to(device),
            counts.to(device),
            offsets.to(device),
            edges.to(device),
            fixed_count,
            all(p == list(range(nq)) for p in q_perms),
            all(p == list(range(num_kv_blocks)) for p in kv_perms),
            bucket_size_blocks,
            edges64,
        )


def _validate(q, k, v, grouping, softmax_scale):
    if not isinstance(grouping, _GroupedMetadata):
        raise TypeError("grouping must be returned by block_sparse_attention_grouped_prepare")
    b, h, nq = grouping.q_permutation.shape
    nk = grouping.kv_permutation.shape[2]
    for name, tensor, shape in (("q", q, (b, nq * 64, h, 128)), ("k", k, (b, nk * 64, h, 128)), ("v", v, (b, nk * 64, h, 128))):
        _validate_tensor(name, tensor, shape, torch.bfloat16, grouping.q_permutation.device)
    if torch.cuda.get_device_capability(q.device) not in {(10, 0), (10, 3)}:
        raise NotImplementedError("grouped BSA currently supports SM100 and SM103")
    if softmax_scale is not None and not isinstance(softmax_scale, (int, float)):
        raise ValueError("softmax_scale must be a Python number")
    scale = 128**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    from cudnn.frost.buffers import cutedsl_requirement_error

    error = cutedsl_requirement_error("grouped block sparse attention")
    if error:
        raise RuntimeError(error)
    return scale


def _validate_tensor(name, tensor, shape, dtype, device):
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype or tensor.device != device:
        raise ValueError(f"{name} must have shape {tuple(shape)}, dtype {dtype}, and device {device}")
    if not tensor.is_contiguous() or tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be contiguous and 16-byte aligned")


def _permute(tensor, permutation, identity, scatter=False):
    if identity:
        return tensor
    from ._grouped_reorder import reorder

    result = torch.empty_like(tensor)
    reorder(tensor, result, permutation, scatter=scatter)
    return result


def _record_inputs(grouping, *tensors):
    from cudnn._torch_stream import record_streams

    record_streams(
        (
            *tensors,
            grouping.q_permutation,
            grouping.kv_permutation,
            grouping.q2k_block_index,
            grouping.q2k_block_nums,
            grouping.bucketed_k2q_offsets,
            grouping.bucketed_k2q_indices,
        ),
        None,
        grouping.q_permutation.device,
    )


def forward(q, k, v, grouping, softmax_scale):
    scale = _validate(q, k, v, grouping, softmax_scale)
    # No implicit grouping in the ordinary API: these permutations are the
    # explicitly requested operation, including the return to original order.
    with torch.cuda.device(q.device):
        from ._interface import bsa_attn_fwd

        _record_inputs(grouping, q, k, v)
        qp = _permute(q, grouping.q_permutation, grouping.identity_q)
        kp = _permute(k, grouping.kv_permutation, grouping.identity_kv)
        vp = _permute(v, grouping.kv_permutation, grouping.identity_kv)
        o, lse = bsa_attn_fwd(
            qp,
            kp,
            vp,
            grouping.q2k_block_index,
            grouping.fixed_count or grouping.q2k_block_index.shape[-1],
            q2k_block_nums=None if grouping.fixed_count is not None else grouping.q2k_block_nums,
            softmax_scale=scale,
            sparse_block_size=128,
            layout="bshd",
            return_lse=True,
            allow_empty_block_nums=False,
        )
        return TupleDict(
            o_tensor=_permute(o, grouping.q_permutation, grouping.identity_q, scatter=True),
            lse_tensor=_permute(lse, grouping.q_permutation, grouping.identity_q, scatter=True),
        )


def backward(dout, q, k, v, out, lse, grouping, softmax_scale):
    scale = _validate(q, k, v, grouping, softmax_scale)
    _validate_tensor("out", out, q.shape, q.dtype, q.device)
    _validate_tensor("dout", dout, q.shape, q.dtype, q.device)
    _validate_tensor("lse", lse, (q.shape[0], q.shape[2], q.shape[1]), torch.float32, q.device)
    with torch.cuda.device(q.device):
        import tvm_ffi
        from .csrc.bwd.sm100_blk128.bsa_bwd_sm100 import bsa_sm100_blk128_bwd_bucketed_k2q_csr

        _record_inputs(grouping, q, k, v, out, lse, dout)
        qp, op, dop = [_permute(t, grouping.q_permutation, grouping.identity_q) for t in (q, out, dout)]
        lp = _permute(lse, grouping.q_permutation, grouping.identity_q)
        kp, vp = [_permute(t, grouping.kv_permutation, grouping.identity_kv) for t in (k, v)]
        # This native entry resets all accumulators on every call, including
        # graph replay. Its CSR metadata was built once during preparation.
        with tvm_ffi.use_torch_stream():
            dq, dk, dv = bsa_sm100_blk128_bwd_bucketed_k2q_csr(
                dop.transpose(1, 2),
                qp.transpose(1, 2),
                kp.transpose(1, 2),
                vp.transpose(1, 2),
                op.transpose(1, 2),
                lp,
                grouping.bucketed_k2q_offsets,
                grouping.bucketed_k2q_indices,
                softmax_scale=scale,
            )
        return TupleDict(
            dq_tensor=_permute(dq.transpose(1, 2), grouping.q_permutation, grouping.identity_q, scatter=True),
            dk_tensor=_permute(dk.transpose(1, 2), grouping.kv_permutation, grouping.identity_kv, scatter=True),
            dv_tensor=_permute(dv.transpose(1, 2), grouping.kv_permutation, grouping.identity_kv, scatter=True),
        )
