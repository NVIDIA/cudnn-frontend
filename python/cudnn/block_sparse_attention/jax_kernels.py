# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XLA stream adapters for the existing SM100 blk128 kernels."""

import cutlass
import cutlass.cute as cute
from .csrc.fwd.sm100_blk128.bsa_fwd_sm100 import BlockSparseAttnForwardSm100Blk128
from .csrc.bwd.bucketed_k2q_csr import BucketedK2QCsrUniversal
from .csrc.bwd.bsa_bwd_preprocess import BlockSparseAttnBackwardPreprocess
from .csrc.bwd.bsa_bwd_postprocess import BlockSparseAttnBackwardPostprocess
from .csrc.bwd.sm100_blk128.bsa_bwd_sm100 import BlockSparseAttnBackwardSm100Blk128, BsaK2qCsrTensors


class Forward:
    def __init__(self, head_dim, count, variable, allow_empty, scale):
        self.kernel = BlockSparseAttnForwardSm100Blk128(head_dim, allow_empty_block_nums=allow_empty, has_block_sizes=False)
        self.count = count
        self.variable = variable
        self.scale = scale

    @cute.jit
    def __call__(self, stream, q, k, v, indices, nums, o, lse):
        self.kernel(q, k, v, o, lse, self.scale, indices, None, self.count, nums if cutlass.const_expr(self.variable) else None, stream)


class Backward:
    def __init__(self, head_dim, count, variable, capacity, bucket_size, groups, scale):
        self.csr = BucketedK2QCsrUniversal(count, bucket_size, variable, capacity)
        self.pre = BlockSparseAttnBackwardPreprocess(cutlass.BFloat16, head_dim, head_dim)
        self.main = BlockSparseAttnBackwardSm100Blk128(head_dim, force_dkv_postprocess=groups > 1)
        self.post = BlockSparseAttnBackwardPostprocess(cutlass.BFloat16, head_dim, 100)
        self.groups = groups
        self.scale = scale

    @cute.jit
    def __call__(
        self,
        stream,
        q,
        k,
        v,
        do,
        o,
        lse,
        indices,
        nums,
        dq,
        dk,
        dv,
        counts,
        local,
        totals,
        offsets,
        cursors,
        edges,
        dpsum,
        log2lse,
        dqacc,
        dkacc,
        dvacc,
    ):
        self.csr(counts, local, totals, offsets, cursors, edges, indices, nums, stream)
        self.pre(o, do, dpsum, lse, log2lse, dqacc, None, None, None, stream)
        if cutlass.const_expr(self.groups > 1):
            self.main(q, k, v, do, log2lse, dpsum, dqacc, dkacc, dvacc, self.scale, BsaK2qCsrTensors(offsets, edges), stream)
        else:
            self.main(q, k, v, do, log2lse, dpsum, dqacc, dk, dv, self.scale, BsaK2qCsrTensors(offsets, edges), stream)
        self.post(dqacc, dq, self.scale, None, None, stream)
        if cutlass.const_expr(self.groups > 1):
            self.post(dkacc, dk, self.scale, None, None, stream)
            self.post(dvacc, dv, 1.0, None, None, stream)
