import cutlass
import cutlass.cute as cute

from cudnn.deepseek_sparse_attention.utils.sm90 import primitives as sm90_ops


class PackGQA:
    def __init__(
        self,
        m_block_size: cutlass.Constexpr[int],
        head_dim_padded: cutlass.Constexpr[int],
        check_hdim_oob: cutlass.Constexpr[bool],
        qhead_per_kvhead: cutlass.Constexpr[int],
    ):
        self.m_block_size = m_block_size
        self.head_dim_padded = head_dim_padded
        self.check_hdim_oob = check_hdim_oob
        self.qhead_per_kvhead = qhead_per_kvhead

    @cute.jit
    def compute_ptr(
        self,
        tensor: cute.Tensor,
        cRows: cute.Tensor,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        threads_per_row: cutlass.Constexpr[int],
        num_threads: cutlass.Constexpr[int],
    ):
        """Compute per-thread pointers for packed layout ((qhpkv, seqlen_q),).
        Matches forward flash_attn.cute.pack_gqa.PackGQA.compute_ptr exactly.
        """
        num_ptr_per_thread = cute.ceil_div(cute.size(cRows), threads_per_row)
        tPrPtr = cute.make_rmem_tensor(num_ptr_per_thread, cutlass.Int64)
        for i in cutlass.range_constexpr(num_ptr_per_thread):
            row = i * num_threads + cRows[tidx % threads_per_row][0]
            idx = block * self.m_block_size + row
            m_idx = idx // self.qhead_per_kvhead
            h_idx = idx - m_idx * self.qhead_per_kvhead
            tPrPtr[i] = sm90_ops.elem_pointer(tensor, ((h_idx, m_idx),)).toint()
        return tPrPtr

    @cute.jit
    def store_O(
        self,
        mO: cute.Tensor,  # ((qhead_per_kvhead, seqlen_q), headdim)
        tOrO: cute.Tensor,  # (m_block_size, head_dim_padded) split across threads
        gmem_tiled_copy: cute.TiledCopy,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tOcO = gmem_thr_copy.partition_S(cO)
        t0OcO = gmem_thr_copy.get_slice(0).partition_S(cO)
        tOpO = sm90_ops.predicate_k(tOcO, limit=mO.shape[1])
        tOcO_row = tOcO[0, None, 0]
        # Use layout_src_tv_tiled for compatibility with the shared tiled-copy helper.
        layout_tv = getattr(gmem_tiled_copy, "layout_src_tv_tiled", gmem_tiled_copy.layout_tv_tiled)
        threads_per_row = layout_tv[0].shape[0]
        assert cute.arch.WARP_SIZE % threads_per_row == 0, "threads_per_row must divide WARP_SIZE"
        num_threads = gmem_tiled_copy.size
        tPrOPtr = self.compute_ptr(mO[None, 0], tOcO_row, tidx, block, threads_per_row, num_threads)
        for m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            o_ptr_i64 = sm90_ops.shuffle_sync(tPrOPtr[m // threads_per_row], m % threads_per_row, width=threads_per_row)
            o_gmem_ptr = cute.make_ptr(mO.element_type, o_ptr_i64, cute.AddressSpace.gmem, assumed_align=16)
            if t0OcO[0, m, 0][0] < seqlen * self.qhead_per_kvhead - block * self.m_block_size - tOcO_row[0][0]:
                mO_cur = cute.make_tensor(o_gmem_ptr, (self.head_dim_padded,))
                elems_per_load = cute.size(tOrO.shape[0][0])
                mO_cur_copy = cute.tiled_divide(mO_cur, (elems_per_load,))
                for k in cutlass.range_constexpr(cute.size(tOrO.shape[2])):
                    ki = tOcO[0, 0, k][1] // elems_per_load
                    cute.copy(
                        gmem_thr_copy,
                        tOrO[None, m, k],
                        mO_cur_copy[None, ki],
                        pred=tOpO[None, m, k] if cutlass.const_expr(self.check_hdim_oob) else None,
                    )

    @cute.jit
    def load_LSE_packed(
        self,
        base_ptr_i64: cutlass.Int64,
        seqlen_q: cutlass.Int32,
        sLSE: cute.Tensor,  # (tile_m,) - SMEM buffer
        m_block: cutlass.Int32,
        tile_m: cutlass.Constexpr[int],
        tidx: cutlass.Int32,  # 0-31 for Warp 0
    ):
        """Load LSE from packed global memory to SMEM. Warp 0 only (32 threads)."""
        qhpkv = self.qhead_per_kvhead
        rows_per_thread = cute.ceil_div(tile_m, cute.arch.WARP_SIZE)
        for i in cutlass.range_constexpr(rows_per_thread):
            row = i * cute.arch.WARP_SIZE + tidx
            if row < tile_m:
                idx = m_block * tile_m + row
                m_idx = idx // qhpkv
                h_idx = idx - m_idx * qhpkv
                ptr = sm90_ops.elem_pointer_packed_i64(
                    base_ptr_i64,
                    h_idx,
                    m_idx,
                    seqlen_q,
                    cutlass.Float32,
                    cute.AddressSpace.gmem,
                )
                gmem_val = cute.make_tensor(ptr, (1,))
                sLSE[row] = gmem_val[0]

    @cute.jit
    def load_dPsum_packed(
        self,
        base_ptr_i64: cutlass.Int64,
        seqlen_q: cutlass.Int32,
        sdPsum: cute.Tensor,  # (tile_m,) - SMEM buffer
        m_block: cutlass.Int32,
        tile_m: cutlass.Constexpr[int],
        tidx: cutlass.Int32,  # 0-31 for Warp 0
    ):
        """Load dPsum from packed global memory to SMEM. Warp 0 only (32 threads)."""
        qhpkv = self.qhead_per_kvhead
        rows_per_thread = cute.ceil_div(tile_m, cute.arch.WARP_SIZE)
        for i in cutlass.range_constexpr(rows_per_thread):
            row = i * cute.arch.WARP_SIZE + tidx
            if row < tile_m:
                idx = m_block * tile_m + row
                m_idx = idx // qhpkv
                h_idx = idx - m_idx * qhpkv
                ptr = sm90_ops.elem_pointer_packed_i64(
                    base_ptr_i64,
                    h_idx,
                    m_idx,
                    seqlen_q,
                    cutlass.Float32,
                    cute.AddressSpace.gmem,
                )
                gmem_val = cute.make_tensor(ptr, (1,))
                sdPsum[row] = gmem_val[0]

    @cute.jit
    def load_Weights_packed(
        self,
        base_ptr_i64: cutlass.Int64,
        seqlen_q: cutlass.Int32,
        sWeights: cute.Tensor,  # (tile_m,) - SMEM buffer
        m_block: cutlass.Int32,
        tile_m: cutlass.Constexpr[int],
        tidx: cutlass.Int32,  # 0-31 for Warp 0
    ):
        """Load Weights from packed global memory to SMEM. Warp 0 only (32 threads)."""
        qhpkv = self.qhead_per_kvhead
        rows_per_thread = cute.ceil_div(tile_m, cute.arch.WARP_SIZE)
        for i in cutlass.range_constexpr(rows_per_thread):
            row = i * cute.arch.WARP_SIZE + tidx
            if row < tile_m:
                idx = m_block * tile_m + row
                m_idx = idx // qhpkv
                h_idx = idx - m_idx * qhpkv
                ptr = sm90_ops.elem_pointer_packed_i64(
                    base_ptr_i64,
                    h_idx,
                    m_idx,
                    seqlen_q,
                    cutlass.BFloat16,
                    cute.AddressSpace.gmem,  # Weights use bf16.
                )
                gmem_val = cute.make_tensor(ptr, (1,))
                sWeights[row] = gmem_val[0]
