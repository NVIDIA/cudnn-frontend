from typing import Optional

import cutlass
import cutlass.cute as cute

"""
This consolidates all the info related to sequence length. This is so that we can do all
the gmem reads once at the beginning of each tile, rather than having to repeat these reads
to compute various things like n_block_min, n_block_max, etc.
"""

class SeqlenInfo:
    def __init__(
        self,
        batch_idx: cutlass.Int32,
        max_seqlen_q: cutlass.Int32,
        max_seqlen_k: cutlass.Int32,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        num_contexts: Optional[cute.Tensor],
        num_targets: Optional[cute.Tensor],
        page_indptrs: Optional[cute.Tensor] = None,
    ):
        assert cu_seqlens_q is not None and cu_seqlens_k is not None
        self.offset_q = cu_seqlens_q[batch_idx]
        self.offset_k = cu_seqlens_k[batch_idx]
        self.seqlen_q = cu_seqlens_q[batch_idx + 1] - self.offset_q
        self.seqlen_k = cu_seqlens_k[batch_idx + 1] - self.offset_k
        self.seqlen_c = num_contexts[batch_idx] if num_contexts is not None else 0
        self.seqlen_h = self.seqlen_k - num_targets[batch_idx] if num_targets is not None else self.seqlen_k

        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.page_ind = page_indptrs[batch_idx] if page_indptrs is not None else 0
