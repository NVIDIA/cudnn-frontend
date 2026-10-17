# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit, immutable routing analysis for the optional blk64 load64 path.

Preparation reads the sparse graph on the CPU. Execution only checks host
declarations and uses the prepared GPU metadata; it never discovers pairs.
Forward and backward eligibility are independent.
"""

from collections import defaultdict
from dataclasses import dataclass
import math

import torch


def _version(tensor):
    if tensor is None:
        return None
    try:
        return tensor._version
    except RuntimeError:  # Inference tensors have no version counter.
        return None


@dataclass(frozen=True, eq=False)
class Blk64Plan:
    _source: torch.Tensor
    _source_counts: torch.Tensor | None
    _versions: tuple
    _fixed_count: int
    num_kv_blocks: int
    bucket_size_blocks: int
    forward_enabled: bool
    backward_enabled: bool
    forward_reason: str
    backward_reason: str
    _q_permutation: torch.Tensor | None
    _kv_permutation: torch.Tensor | None
    _q_identity: bool
    _kv_identity: bool
    _routes: torch.Tensor | None
    _counts: torch.Tensor | None
    _pair_count: int
    _offsets: torch.Tensor | None
    _edges: torch.Tensor | None

    def check_binding(self, routes, counts, fixed, sparse_block_size):
        if sparse_block_size != 64:
            raise ValueError("blk64_plan requires sparse_block_size=64")
        if routes is not self._source or counts is not self._source_counts or fixed != self._fixed_count:
            raise ValueError("blk64_plan must be used with the same routing tensors and block_sparse_num used at preparation")
        if (_version(routes), _version(counts)) != self._versions:
            raise ValueError("sparse metadata changed; rebuild blk64_plan before execution")


def _pairs(rows):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[tuple(row)].append(i)
    if any(len(group) % 2 for group in groups.values()):
        return None
    # Prefer the identity permutation when each physical pair is already exact.
    if all(rows[i] == rows[i + 1] for i in range(0, len(rows), 2)):
        return list(range(len(rows)))
    return [i for group in groups.values() for i in group]


def prepare(routes, fixed, nk, counts, bucket):
    if not isinstance(routes, torch.Tensor) or routes.ndim != 4 or routes.dtype != torch.int32 or not routes.is_cuda:
        raise ValueError("q2k_block_index must be a rank-4 CUDA int32 tensor")
    b, h, nq, capacity = routes.shape
    if min(b, h, nq, capacity) < 1 or type(nk) is not int or nk < 1:
        raise ValueError("routing dimensions and num_kv_blocks must be positive")
    if max(nq, nk) * 64 >= 2**31:
        raise ValueError("blk64 plan exceeds int32 token indexing")
    fixed = capacity if fixed is None else fixed
    if type(fixed) is not int or not 1 <= fixed <= capacity:
        raise ValueError("block_sparse_num must be a positive integer within routing capacity")
    # Units match the ordinary blk64 backward API: Q64 blocks, not Q128.
    bucket = 1024 if bucket is None else bucket
    if type(bucket) is not int or bucket < 1:
        raise ValueError("bucket_size_blocks must be a positive number of Q64 blocks")
    if counts is not None and (
        not isinstance(counts, torch.Tensor) or counts.dtype != torch.int32 or counts.shape != (b, h, nq) or counts.device != routes.device
    ):
        raise ValueError("q2k_block_nums must be int32 [B,H,Q64] on the routing device")

    with torch.cuda.device(routes.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("build blk64_plan before CUDA Graph capture")
        host = routes.cpu().tolist()
        nums = counts.cpu().tolist() if counts is not None else None
        qp_all, kp_all, frows_all, offsets_all, edges_all = [], [], [], [], []
        fwd_ok, bwd_ok = nq % 2 == 0 and nk % 2 == 0, nq % 2 == 0 and nk % 2 == 0
        fwd_reason = bwd_reason = "requires full even Q64 and KV64 block counts"
        nb = (nq + bucket - 1) // bucket
        for bi in range(b):
            for hi in range(h):
                rows, reverse = [], [[] for _ in range(nk)]
                for qi, ids in enumerate(host[bi][hi]):
                    length = nums[bi][hi][qi] if nums is not None else fixed
                    if not 0 <= length <= capacity:
                        raise ValueError("q2k_block_nums values must be within routing capacity")
                    row = tuple(sorted(ids[:length]))
                    if any(k < 0 or k >= nk for k in row) or len(set(row)) != length:
                        raise ValueError("valid routing prefixes must contain unique in-range KV block indices")
                    rows.append(row)
                    for ki in row:
                        reverse[ki].append(qi)
                qp, kp = _pairs(rows), _pairs(reverse)
                head_fwd = qp is not None and all(len(row) > 0 and len(row) % 2 == 0 for row in rows)
                head_bwd = kp is not None and all(rows)
                if not head_fwd:
                    fwd_reason = "Q64 neighborhoods must pair exactly and have positive even lengths"
                if not head_bwd:
                    bwd_reason = "KV64 reverse neighborhoods must pair exactly; Q rows must be nonempty"
                fwd_ok &= head_fwd
                bwd_ok &= head_bwd
                qp_all.append(qp)
                kp_all.append(kp)
                frows_all.append([rows[qp[i]] for i in range(0, nq, 2)] if head_fwd else [])
                if head_bwd:
                    owners = [[[] for _ in range(nk // 2)] for _ in range(nb)]
                    for ki in range(nk // 2):
                        for qi in reverse[kp[2 * ki]]:
                            owners[qi // bucket][ki].append(qi)
                    offsets, packed = [], []
                    for group in owners:
                        off = [len(packed)]
                        for row in group:
                            for i in range(0, len(row), 2):
                                q1 = row[i + 1] if i + 1 < len(row) else nq
                                packed.append(row[i] | (q1 << 32))
                            off.append(len(packed))
                        offsets.append(off)
                    if len(packed) >= 2**31:
                        raise ValueError("paired CSR exceeds int32 edge indexing")
                    offsets_all.append(offsets)
                    edges_all.append(packed)

        def tensor(data, dtype, shape):
            return torch.tensor(data, dtype=dtype, device=routes.device).reshape(shape)

        qp_tensor = kp_tensor = fwd_routes = fwd_counts = offsets_tensor = edges_tensor = None
        pair_count = 0
        qi = ki = True
        if fwd_ok:
            qi = all(p == list(range(nq)) for p in qp_all)
            qp_tensor = tensor(qp_all, torch.int32, (b, h, nq))
            cap = max(len(row) for rows in frows_all for row in rows)
            fwd_routes = tensor([[row + (0,) * (cap - len(row)) for row in rows] for rows in frows_all], torch.int32, (b, h, nq // 2, cap))
            pair_count = cap // 2
            lengths = [[len(row) // 2 for row in rows] for rows in frows_all]
            # The fixed-count kernel requires an even number of compute tiles.
            if pair_count % 2 or any(n != pair_count for rows in lengths for n in rows):
                fwd_counts = tensor(lengths, torch.int32, (b, h, nq // 2))
        if bwd_ok:
            ki = all(p == list(range(nk)) for p in kp_all)
            kp_tensor = tensor(kp_all, torch.int32, (b, h, nk))
            cap = max(1, max(map(len, edges_all)))
            offsets_tensor = tensor(offsets_all, torch.int32, (b, h, nb, nk // 2 + 1))
            edges_tensor = tensor([row + [0] * (cap - len(row)) for row in edges_all], torch.int64, (b, h, cap))
        return Blk64Plan(
            routes,
            counts,
            (_version(routes), _version(counts)),
            fixed,
            nk,
            bucket,
            fwd_ok,
            bwd_ok,
            "exact Q pairs" if fwd_ok else fwd_reason,
            "exact KV pairs" if bwd_ok else bwd_reason,
            qp_tensor,
            kp_tensor,
            qi,
            ki,
            fwd_routes,
            fwd_counts,
            pair_count,
            offsets_tensor,
            edges_tensor,
        )


def eligible(plan, q, k, v, arch, layout, scale, block_sizes, *, backward=False, extra=()):
    """Host-only checks; unsupported declarations take the unchanged native path."""
    if not (plan.backward_enabled if backward else plan.forward_enabled):
        return False
    if arch not in (100, 103) or layout != "bshd" or block_sizes is not None:
        return False
    b, h, nq, _ = plan._source.shape
    if q.shape != (b, nq * 64, h, 128) or k.shape != (b, plan.num_kv_blocks * 64, h, 128) or v.shape != k.shape:
        return False
    if q.dtype != torch.bfloat16 or any(not x.is_contiguous() or x.data_ptr() % 16 for x in (q, k, v, *extra) if x is not None):
        return False
    # Do not silently reinterpret the native scale contract.
    if scale is not None and (type(scale) not in (float, int) or not math.isfinite(scale) or scale <= 0):
        return False
    from cudnn.frost.buffers import cutedsl_requirement_error

    error = cutedsl_requirement_error("blk64 paired TMA attention")
    if error:
        raise RuntimeError(error)
    return True
