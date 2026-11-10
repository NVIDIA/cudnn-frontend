# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 blk64 backward with paired KV neighborhoods.

Matched KV64 pairs (the SM100 paired-KV metadata in ``local_pair``) order each pair's CSR lists as [first-only |
common | second-only]. The blk128 backward in paired mode gives each pair one CTA, one KV64 block per compute WG, and
walks that list once: Q/dO, LSE/dPsum and the dQ reduction are shared, and each WG computes only the Q blocks its own
block attends, so Q/dO are swept once for all edges. A blk128 compute WG's FP32 dK/dV fragment is one blk64
accumulator chunk and the dQ fragments match, so the blk64 preprocess, workspace and postprocess serve it unchanged.

The pairing metadata (``PairedKvMetadataSm90``) and the compute (``MixedKvBackwardSm90``) are two compiled entry
points: the metadata depends on the sparse pattern only, so a caller that reuses one pattern across calls builds it
once (``block_sparse_attention_prepare_backward``) and the backward launches the preprocess, the main kernel and the
postprocess only.

The FP32 dK/dV workspace is not zeroed: the first (pair, Q group) task to reach a KV64 block stores its fragments
and later ones reduce-add (``pair_sync`` tickets, see ``epilogue_paired_kv``); the dK/dV postprocess writes zeros for
blocks no task arrived at. ``pair_sync`` (row 0: arrival tickets, row 1: first-store flags) is per-call state that
the caller hands in zeroed.
"""

from typing import Tuple

import cuda.bindings.driver as cuda

import cutlass.cute as cute
from cutlass import Float32, Int32

from cudnn.block_sparse_attention.csrc.utils.cute_dsl_utils import assume_tensor_aligned
from .bsa_bwd_sm90 import BlockSparseAttnBackwardSm90Blk64
from ..sm90_blk128.bsa_bwd_sm90 import BlockSparseAttnBackwardSm90Blk128


class PairedKvMetadataSm90:
    """Pair KV neighborhoods and partition the K2Q CSR lists of each pair; everything here depends on the mask only."""

    def __init__(self, bucket_size_blocks):
        from ..local_pair import PairedKvNeighborhoods, LocalPairMatch8, LocalPairPartition8
        from ..local_pair_refine_batch_wide import LocalPairRefineBatchWide

        self.bucket_size = bucket_size_blocks
        self.classifier = PairedKvNeighborhoods()
        self.matcher = LocalPairMatch8()
        self.refiner = LocalPairRefineBatchWide(32, 32, 128)
        self.partition = LocalPairPartition8()

    @cute.jit
    def __call__(
        self,
        offsets: cute.Tensor,
        indices: cute.Tensor,
        q2k: cute.Tensor,
        counts: cute.Tensor,
        bits: cute.Tensor,
        duplicates: cute.Tensor,
        flags: cute.Tensor,
        residuals: cute.Tensor,
        tasks: cute.Tensor,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        b, h, _ = counts.shape
        q_blocks = counts.shape[2]
        self.classifier.clear(bits, duplicates).launch(grid=(cute.ceil_div(cute.size(bits), 256), 1, 1), block=(256, 1, 1), stream=stream)
        self.classifier.scatter(q2k, counts, bits, duplicates).launch(grid=(q_blocks, h, b), block=(128, 1, 1), stream=stream)
        owners = cute.make_tensor(
            flags.iterator + bits.shape[2],
            cute.make_layout((b, h, cute.ceil_div(bits.shape[2], 2)), stride=flags.stride),
        )
        self.matcher(bits, duplicates, flags, owners, residuals[None, None, 0, None], stream)
        self.refiner(bits, duplicates, flags, owners, residuals[None, None, 0, None], stream)
        self.partition(bits, flags, offsets, indices, residuals, self.bucket_size, stream, tasks)


class MixedKvBackwardSm90:
    """Run every pair's Q walk in one blk128-shaped kernel over the metadata that ``PairedKvMetadataSm90`` built."""

    def __init__(self, dtype, head_dim, bucket_size_blocks, scale_nonpositive, has_kv_tail):
        self.bucket_size = bucket_size_blocks
        # Supplies the preprocess, the FP32 workspace and the postprocess; its own mainloop does not run.
        self.regular = BlockSparseAttnBackwardSm90Blk64(dtype, head_dim, head_dim)
        self.paired = BlockSparseAttnBackwardSm90Blk128(dtype, head_dim, head_dim, scale_nonpositive, has_kv_tail=has_kv_tail, paired=True)

    @cute.jit
    def __call__(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        mdO: cute.Tensor,
        mO: cute.Tensor,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdQ: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        offsets: cute.Tensor,
        indices: cute.Tensor,
        mBlockSizes: cute.Tensor | None,
        workspace: cute.Tensor,
        softmax_scale: Float32,
        flags: cute.Tensor,
        residuals: cute.Tensor,
        tasks: cute.Tensor,
        sync: cute.Tensor,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # Public BSA tensors are BHSD; the SM90 kernels use BSHD stride views.
        def _bhsd_to_bshd(t):
            return assume_tensor_aligned(cute.make_tensor(t.iterator, cute.select(t.layout, mode=[0, 2, 1, 3])))

        mdO, mO, mQ, mK, mV, mdQ, mdK, mdV = [_bhsd_to_bshd(t) for t in (mdO, mO, mQ, mK, mV, mdQ, mdK, mdV)]
        regular = self.regular
        mdPsum, mLSElog2, mdQaccum, mdKaccum, mdVaccum = regular.get_workspace_tensor(problem_shape, workspace)
        mdPsum, mLSElog2, mdQaccum, mdKaccum, mdVaccum = [assume_tensor_aligned(t) for t in (mdPsum, mLSElog2, mdQaccum, mdKaccum, mdVaccum)]

        regular._preprocess_call(mO, mdO, mdPsum, mLSE, mLSElog2, mdQaccum, stream)
        self.paired._bwd_call(
            mQ,
            mK,
            mV,
            mdO,
            mLSElog2,
            mdPsum,
            mdQaccum,
            mdKaccum,
            mdVaccum,
            softmax_scale,
            mBlockSizes,
            offsets,
            indices,
            flags,
            residuals,
            tasks,
            sync,
            stream=stream,
        )
        arrivals = sync[None, None, 0, None]
        regular._postprocess_call(mdQaccum, mdQ, softmax_scale, stream=stream)
        regular._postprocess_call(mdKaccum, mdK, softmax_scale, arrivals, stream=stream)
        regular._postprocess_call(mdVaccum, mdV, Float32(1.0), arrivals, stream=stream)
