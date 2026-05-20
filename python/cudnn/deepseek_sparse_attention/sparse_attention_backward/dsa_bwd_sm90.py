import math
import operator
from functools import partial
from typing import Callable, Literal, Optional, Tuple, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils import LayoutEnum

from cudnn.deepseek_sparse_attention.utils.copy import (
    load_s2r,
    tiled_copy_1d,
    tiled_copy_2d,
    tma_get_copy_fn,
)
from cudnn.deepseek_sparse_attention.utils.sm90.mma import (
    gemm_w_idx,
    gemm_zero_init,
    make_smem_layout,
    mma_partition_fragment_AB,
)
from cudnn.deepseek_sparse_attention.utils.sm90.bwd_barriers import NamedBarrierBwd
from cudnn.deepseek_sparse_attention.utils.seqlen import SeqlenInfoQK
from cudnn.deepseek_sparse_attention.utils.sm90.bwd_tile_scheduler import (
    ParamsBase,
    SingleTileScheduler,
    TileSchedulerArguments,
)
from cudnn.deepseek_sparse_attention.utils.sm90.primitives import (
    atomic_add_fp32,
    atomic_add_fp32x4,
    convert_layout_acc_frgA,
    cvt_f16,
    get_smem_store_atom,
    make_acc_tensor_frgA_view,
    make_acc_tensor_mn_view,
    predicate_k,
    select,
    transpose_view,
    warp_reduce,
)


@dsl_user_op
def _elem_pointer_packed_mh_i64(
    base_ptr_i64: cute.typing.Int,
    h_idx: cute.typing.Int,
    m_idx: cute.typing.Int,
    num_head: cute.typing.Int,
    elem_type: type,
    memspace: cute.AddressSpace,
    *,
    loc=None,
    ip=None,
) -> cute.Pointer:
    linear_idx = cutlass.Int64(m_idx) * cutlass.Int64(num_head) + cutlass.Int64(h_idx)
    byte_offset = linear_idx * elem_type.width // 8
    return cute.make_ptr(
        elem_type,
        cutlass.Int64(base_ptr_i64) + byte_offset,
        memspace,
        assumed_align=16,
    )


@cute.jit
def _load_f32_packed_mh_to_smem(
    base_ptr_i64: cutlass.Int64,
    dst: cute.Tensor,
    m_block: cutlass.Int32,
    tile_m: cutlass.Constexpr[int],
    tidx: cutlass.Int32,
    num_threads: cutlass.Constexpr[int],
    qhead_per_kvhead: cutlass.Constexpr[int],
    num_head: cutlass.Constexpr[int],
):
    rows_per_thread = cute.ceil_div(tile_m, num_threads)
    for i in cutlass.range_constexpr(rows_per_thread):
        row = i * num_threads + tidx
        if row < tile_m:
            idx = m_block * tile_m + row
            m_idx = idx // qhead_per_kvhead
            h_idx = idx - m_idx * qhead_per_kvhead
            ptr = _elem_pointer_packed_mh_i64(
                base_ptr_i64,
                h_idx,
                m_idx,
                num_head,
                cutlass.Float32,
                cute.AddressSpace.gmem,
            )
            gmem_val = cute.make_tensor(ptr, (1,))
            dst[row] = gmem_val[0]


class _FlashAttentionDSABackwardPreprocessSm90:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        arch: Literal[80, 90, 100],
        m_block_size: int = 128,
        num_threads: int = 128,
    ):
        self.dtype = dtype
        self.m_block_size = m_block_size
        self.arch = arch
        # padding head_dim to a multiple of 32 as k_block_size
        hdim_multiple_of = 32
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.num_threads = num_threads

    @staticmethod
    def can_implement(dtype, head_dim, m_block_size, num_threads) -> bool:
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if num_threads < m_block_size:  # For multiplying lse with log2
            return False
        return True

    def _setup_attributes(self):
        # kBlockKGmem must be power of 2 for warp-level summing
        gmem_k_block_size = (
            128 if self.head_dim_padded % 128 == 0 else (64 if self.head_dim_padded % 64 == 0 else (32 if self.head_dim_padded % 32 == 0 else 16))
        )
        self.gmem_tiled_copy_O = tiled_copy_2d(self.dtype, gmem_k_block_size, self.num_threads)
        universal_copy_bits = 128
        num_copy_elems_dQaccum = universal_copy_bits // Float32.width
        assert (self.m_block_size * self.head_dim_padded // num_copy_elems_dQaccum) % self.num_threads == 0
        self.gmem_tiled_copy_dQaccum = tiled_copy_1d(Float32, self.num_threads, num_copy_elems_dQaccum)

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mdPsum: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSElog2: Optional[cute.Tensor],
        mAttnSink: Optional[cute.Tensor],
        mdSink: Optional[cute.Tensor],
        mdQaccum: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        # Get the data type and check if it is fp16 or bf16
        if cutlass.const_expr(not (mO.element_type == mdO.element_type)):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(mO.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if cutlass.const_expr(mdPsum.element_type not in [Float32]):
            raise TypeError("dPsum tensor must be Float32")
        if cutlass.const_expr(mdQaccum is not None):
            if cutlass.const_expr(mdQaccum.element_type not in [Float32]):
                raise TypeError("dQaccum tensor must be Float32")
        if cutlass.const_expr(mLSE is not None):
            assert mLSElog2 is not None, "If mLSE is provided, mLSElog2 must also be provided"
            if cutlass.const_expr(mLSE.element_type not in [Float32]):
                raise TypeError("LSE tensor must be Float32")
            if cutlass.const_expr(mLSElog2.element_type not in [Float32]):
                raise TypeError("LSElog2 tensor must be Float32")
        if cutlass.const_expr(mAttnSink is not None):
            if cutlass.const_expr(mAttnSink.element_type not in [Float32]):
                raise TypeError("attn_sink tensor must be Float32")
        if cutlass.const_expr(mdSink is not None):
            if cutlass.const_expr(mdSink.element_type not in [Float32]):
                raise TypeError("dSink tensor must be Float32")

        # Assume all strides are divisible by 128 bits except the last stride
        new_stride = lambda t: (
            *(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )
        mO, mdO, mdQaccum = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t))) if t is not None else None for t in (mO, mdO, mdQaccum)
        ]

        self._setup_attributes()

        TileScheduler = SingleTileScheduler
        num_head = mO.shape[2]
        num_batch = mO.shape[0]

        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(mO.shape[1], self.m_block_size),
            num_head=num_head,
            num_batch=num_batch,
            num_splits=1,
            seqlen_k=0,
            headdim=0,
            headdim_v=mO.shape[2],
            total_q=mO.shape[0],
            tile_shape_mn=(self.m_block_size, 1),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        self.kernel(
            mO,
            mdO,
            mdPsum,
            mLSE,
            mLSElog2,
            mAttnSink,
            mdSink,
            mdQaccum,
            mCuSeqlensQ,
            mSeqUsedQ,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_dQaccum,
            tile_sched_params,
            TileScheduler,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mdPsum: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSElog2: Optional[cute.Tensor],
        mAttnSink: Optional[cute.Tensor],
        mdSink: Optional[cute.Tensor],
        mdQaccum: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx

        if work_tile.is_valid_tile:
            # ///////////////////////////////////////////////////////////////////////////////
            # Get the appropriate tiles for this thread block.
            # ///////////////////////////////////////////////////////////////////////////////
            seqlen = SeqlenInfoQK.create(
                batch_idx,
                mO.shape[1],
                0,
                mCuSeqlensQ=mCuSeqlensQ,
                mCuSeqlensK=None,
                mSeqUsedQ=mSeqUsedQ,
                mSeqUsedK=None,
            )

            if cutlass.const_expr(not seqlen.has_cu_seqlens_q):
                mO_cur = mO[batch_idx, None, head_idx, None]
                mdO_cur = mdO[batch_idx, None, head_idx, None]
                mdPsum_cur = mdPsum[batch_idx, None, head_idx]
                headdim_v = mO.shape[3]
            else:
                mO_cur = cute.domain_offset((seqlen.offset_q, 0), mO[None, head_idx, None])
                mdO_cur = cute.domain_offset((seqlen.offset_q, 0), mdO[None, head_idx, None])

                padded_offset_q = seqlen.offset_q + batch_idx * self.m_block_size
                if cutlass.const_expr(self.arch >= 90):
                    padded_offset_q = padded_offset_q // self.m_block_size * self.m_block_size
                mdPsum_cur = cute.domain_offset((padded_offset_q,), mdPsum[head_idx, None])
                headdim_v = mO.shape[2]

            blkOdO_shape = (self.m_block_size, self.head_dim_padded)
            # (m_block_size, head_dim)
            gO = cute.local_tile(mO_cur, blkOdO_shape, (m_block, 0))
            gdO = cute.local_tile(mdO_cur, blkOdO_shape, (m_block, 0))

            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            # (CPY_Atom, CPY_M, CPY_K)
            tOgO = gmem_thr_copy_O.partition_S(gO)
            tOgdO = gmem_thr_copy_O.partition_S(gdO)

            # ///////////////////////////////////////////////////////////////////////////////
            # Predicate: Mark indices that need to copy when problem_shape isn't a multiple
            # of tile_shape
            # ///////////////////////////////////////////////////////////////////////////////
            # Construct identity layout for KV
            cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_thr_copy_O.get_slice(0).partition_S(cO)
            tOpO = predicate_k(tOcO, limit=headdim_v)
            tOpdO = predicate_k(tOcO, limit=headdim_v)

            seqlen_q = seqlen.seqlen_q
            seqlen_q_rounded = cute.round_up(seqlen_q, self.m_block_size)

            if cutlass.const_expr(mLSE is not None):
                if cutlass.const_expr(not seqlen.has_cu_seqlens_q):
                    mLSE_cur = mLSE[batch_idx, None, head_idx]
                else:
                    mLSE_cur = cute.domain_offset((seqlen.offset_q,), mLSE[head_idx, None])

                gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_block,))
                lse = Float32.inf
                if tidx < seqlen_q - m_block * self.m_block_size:
                    lse = gLSE[tidx]

            tOrO = cute.make_rmem_tensor_like(tOgO)
            tOrdO = cute.make_rmem_tensor_like(tOgdO)
            assert cute.size(tOgO, mode=[0]) == cute.size(tOgdO, mode=[0])
            assert cute.size(tOgO, mode=[1]) == cute.size(tOgdO, mode=[1])
            assert cute.size(tOgO, mode=[2]) == cute.size(tOgdO, mode=[2])
            for m in cutlass.range(cute.size(tOrO.shape[1]), unroll_full=True):
                # Instead of using tOcO, we using t0OcO and subtract the offset from the limit
                # (seqlen_q - m_block * kBlockM). This is because the entries of t0OcO are known at compile time.
                if t0OcO[0, m, 0][0] < seqlen_q - m_block * self.m_block_size - tOcO[0][0]:
                    cute.copy(
                        gmem_thr_copy_O,
                        tOgO[None, m, None],
                        tOrO[None, m, None],
                        pred=tOpO[None, m, None] if cutlass.const_expr(self.check_hdim_oob) else None,
                    )
                    cute.copy(
                        gmem_thr_copy_O,
                        tOgdO[None, m, None],
                        tOrdO[None, m, None],
                        pred=tOpdO[None, m, None] if cutlass.const_expr(self.check_hdim_oob) else None,
                    )
            # Sum across the "k" dimension
            dpsum = (tOrO.load().to(Float32) * tOrdO.load().to(Float32)).reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, 1))
            threads_per_row = gmem_tiled_copy_O.layout_src_tv_tiled[0].shape[0]
            assert cute.arch.WARP_SIZE % threads_per_row == 0
            dpsum = warp_reduce(dpsum, operator.add, width=threads_per_row)
            dP_sum = cute.make_rmem_tensor(cute.size(tOrO, mode=[1]), Float32)
            dP_sum.store(dpsum)

            # Write dPsum from rmem -> gmem
            gdPsum = cute.local_tile(mdPsum_cur, (self.m_block_size,), (m_block,))
            # Only the thread corresponding to column 0 writes out the dPsum to gmem
            if tOcO[0, 0, 0][1] == 0:
                for m in cutlass.range(cute.size(dP_sum), unroll_full=True):
                    row = tOcO[0, m, 0][0]
                    row_valid = row < seqlen_q - m_block * self.m_block_size
                    gdPsum[row] = dP_sum[m] if row_valid else 0.0

                    if cutlass.const_expr(mdSink is not None):
                        lse_row = gLSE[row] if row_valid else Float32.inf
                        LOG2_E = math.log2(math.e)
                        lse_log2 = lse_row * LOG2_E
                        sink_log2 = mAttnSink[head_idx] * LOG2_E
                        lse_max_log2 = cute.arch.fmax(lse_log2, sink_log2)
                        sum_exp2 = Float32(cute.math.exp2(lse_log2 - lse_max_log2) + cute.math.exp2(sink_log2 - lse_max_log2))
                        lse_with_sink_log2 = lse_max_log2 + cute.math.log2(sum_exp2)
                        p_sink = cute.math.exp2(sink_log2 - lse_with_sink_log2)
                        if lse_row == Float32.inf:
                            p_sink = Float32(0.0)
                        if row_valid:
                            atomic_add_fp32(
                                -p_sink * dP_sum[m],
                                mdSink.iterator + head_idx,
                            )

            # Clear dQaccum
            if cutlass.const_expr(mdQaccum is not None):
                if cutlass.const_expr(not seqlen.has_cu_seqlens_q):
                    mdQaccum_cur = mdQaccum[batch_idx, head_idx, None]
                else:
                    mdQaccum_cur = cute.domain_offset((padded_offset_q * self.head_dim_padded,), mdQaccum[head_idx, None])

                    # HACK: Compiler doesn't seem to recognize that padding
                    # by padded_offset_q * self.head_dim_padded keeps alignment
                    # since statically divisible by 4

                    mdQaccum_cur_ptr = cute.make_ptr(
                        dtype=mdQaccum_cur.element_type,
                        value=mdQaccum_cur.iterator.toint(),
                        mem_space=mdQaccum_cur.iterator.memspace,
                        assumed_align=mdQaccum.iterator.alignment,
                    )
                    mdQaccum_cur = cute.make_tensor(mdQaccum_cur_ptr, mdQaccum_cur.layout)

                blkdQaccum_shape = (self.m_block_size * self.head_dim_padded,)
                gdQaccum = cute.local_tile(mdQaccum_cur, blkdQaccum_shape, (m_block,))
                gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_slice(tidx)
                tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum)
                zero = cute.make_rmem_tensor_like(tdQgdQaccum)
                zero.fill(0.0)
                cute.copy(gmem_tiled_copy_dQaccum, zero, tdQgdQaccum)

            if cutlass.const_expr(mLSE is not None):
                if cutlass.const_expr(not seqlen.has_cu_seqlens_q):
                    mLSElog2_cur = mLSElog2[batch_idx, None, head_idx]
                else:
                    mLSElog2_cur = cute.domain_offset((padded_offset_q,), mLSElog2[head_idx, None])

                gLSElog2 = cute.local_tile(mLSElog2_cur, (self.m_block_size,), (m_block,))
                LOG2_E = math.log2(math.e)
                if tidx < seqlen_q_rounded - m_block * self.m_block_size:
                    lse_log2 = lse * LOG2_E if lse != -Float32.inf else 0.0
                    if cutlass.const_expr(mAttnSink is not None):
                        sink_log2 = mAttnSink[head_idx] * LOG2_E
                        lse_max_log2 = cute.arch.fmax(lse_log2, sink_log2)
                        sum_exp2 = Float32(cute.math.exp2(lse_log2 - lse_max_log2) + cute.math.exp2(sink_log2 - lse_max_log2))
                        lse_log2 = lse_max_log2 + cute.math.log2(sum_exp2)
                        if lse == Float32.inf:
                            lse_log2 = Float32.inf
                    gLSElog2[tidx] = lse_log2


class FlashAttentionDSABackwardSm90:
    arch = 90

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 64,
        tile_m: int = 64,
        tile_n: int = 64,
        KV_stage: int = 1,
        PdS_stage: int = 1,
        SdP_swapAB: bool = False,
        dKV_swapAB: bool = False,
        dQ_swapAB: bool = False,
        num_threads: int = 384,
        have_topk_length: bool = True,
        max_topk: int = 0,
    ):
        self.dtype = dtype
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        assert qhead_per_kvhead > 1, "This version only supports MQA/GQA (qhead_per_kvhead > 1)"
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = False
        self.is_local = False

        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_threads = num_threads
        self.KV_stage = KV_stage
        self.PdS_stage = PdS_stage
        self.SdP_swapAB = SdP_swapAB
        self.dKV_swapAB = dKV_swapAB
        self.dQ_swapAB = dQ_swapAB
        self.have_topk_length = have_topk_length
        self.max_topk = max_topk
        # wg0 is consumer and producer, wg1 is consumer
        self.num_mma_warp_groups = self.num_threads // 128

        self.hdim_chunk = min(64, self.tile_hdim)  # hdim chunking for dKV GEMM3/5
        self.N_hdim_chunks = self.tile_hdim // self.hdim_chunk
        self.hdim_chunk_dq = min(128, self.tile_hdim)  # hdim chunking for dQ GEMM4 (each WG does 2 quarter GEMMs)
        self.N_dQ_chunks = self.tile_hdim // self.hdim_chunk_dq
        # WG0 always does 128+128 (same as V3.1, no extra register pressure).
        # WG1 absorbs the remainder: 128+128 for d=512, 128+192 for d=576.
        self.hdim_chunk_dq_wg0_1 = self.hdim_chunk_dq  # always 128
        self.hdim_chunk_dq_wg1_1 = self.tile_hdim - self.hdim_chunk_dq * 3 if not self.same_hdim_kv else self.hdim_chunk_dq

        # K=V or V=K[0:d_v] constraints
        assert self.tile_hdim >= self.tile_hdimv, "tile_hdim must be >= tile_hdimv"
        assert self.num_mma_warp_groups == 2, "WG-split mode requires 2 MMA warp groups"
        assert self.tile_hdim % self.hdim_chunk == 0

    @staticmethod
    def can_implement(dtype, head_dim, head_dim_v, tile_m, tile_n, KV_stage, num_threads) -> bool:
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0 or head_dim_v % 8 != 0:
            return False
        if tile_n % 16 != 0 or num_threads % 32 != 0:
            return False
        return True

    def _check_type(self, mQ_type, mKV_type, mdO_type, mLSE_type, mdPsum_type, mdQ_type, mdKV_type):
        if const_expr(not (mQ_type == mKV_type == mdO_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mLSE_type not in [Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mdPsum_type not in [Float32]):
            raise TypeError("dPsum tensor must be Float32")
        if const_expr(mdQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("mdQ tensor must be Float16 or BFloat16")
        if const_expr(mdKV_type != Float32):
            raise TypeError("mdKVaccum tensor must be Float32")
        assert mQ_type == self.dtype

    def _setup_attributes(self):
        # sQ/sdO single-stage, sKV single-buffer, sPdS for P/dS
        self.sQ_layout, self.sKV_layout, self.sdO_layout, self.sPdS_layout = [
            make_smem_layout(self.dtype, LayoutEnum.ROW_MAJOR, shape, stage)
            for shape, stage in [
                ((self.tile_m, self.tile_hdim), None),
                ((self.tile_n, self.tile_hdim), None),
                ((self.tile_m, self.tile_hdimv), None),
                ((self.tile_m, self.tile_n), self.PdS_stage),
            ]
        ]
        # Quarter-width layout for GEMM4 B operand (sKV columns split into quarters)
        self.sKV_quarter_layout = make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_n, self.hdim_chunk_dq),
            None,  # (tile_n, 128)
        )
        # 192-col layout for WG1 G4_half_3 B operand (576 config: sKV[384:576])
        self.sKV_192_layout = make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_n, self.hdim_chunk_dq_wg1_1),
            None,  # (tile_n, 192 or 128)
        )
        # V subview layout: (tile_n, tile_hdimv) for GEMM2 B operand (V = K[0:d_v])
        self.sV_layout = make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_n, self.tile_hdimv),
            None,  # (tile_n, 512)
        )
        # Half-width layout for epilogue TMA S2G (256 cols at a time)
        self.sQ_half_layout = make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, 256),
            None,
        )
        # 64-col layout for WG1 epilogue tail TMA write (d=576: cols 512:576)
        self.sQ_64_layout = make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, 64),
            None,
        )
        # Quarter-width layout for epilogue R2S target (each acc_dQ writes 128-col slice)
        self.sQ_quarter_layout = make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.hdim_chunk_dq),
            None,  # (tile_m, 128)
        )
        # chunk layouts for GEMM3/5 B operands (64-col slices of sdO/sQ)
        self.sdO_chunk_layout = make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.hdim_chunk),
            None,
        )
        self.sQ_chunk_layout = make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.hdim_chunk),
            None,
        )
        # dKV uses AtomicAdd float4
        # r2s_tiled_copy for gmem partitioning
        self.r2s_tiled_copy_dKVatomic = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
            cute.make_layout((self.num_threads_per_warp_group, 1)),
            cute.make_layout(128 // Float32.width),
        )

    def _get_tiled_mma(self):
        # All 4 tiled_mma use atom_layout_mnk=(1,1,1) — single WG each
        # SdP: S = Q @ KV^T, dP = dO @ KV^T (SS, WG0 only)
        tiled_mma_SdP = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(self.tile_m, self.tile_n),
        )
        # dKV: P^T @ dO_chunk + dS^T @ Q_chunk (SS, WG1)
        # tiler N = hdim_chunk (64), loops over 8 chunks
        tiled_mma_dKV = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            cute.nvgpu.OperandMajorMode.MN,
            cute.nvgpu.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(self.tile_n, self.hdim_chunk),
        )
        # dQ_wg0: dS_scaled(reg) @ KV^T_quarter (RS, WG0) — 2 quarter GEMMs per n_block
        tiled_mma_dQ_wg0 = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(self.tile_m, self.hdim_chunk_dq),
            a_source=warpgroup.OperandSource.RMEM,  # RS GEMM
        )
        # dQ_wg1: sdS @ KV^T_quarter (SS, WG1) — G4_half_2 (128-col)
        tiled_mma_dQ_wg1 = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(self.tile_m, self.hdim_chunk_dq),
        )
        # dQ_wg1_192: sdS @ KV^T_192 (SS, WG1) — G4_half_3 for 576 config (192-col)
        tiled_mma_dQ_wg1_192 = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(self.tile_m, self.hdim_chunk_dq_wg1_1),  # (tile_m, 192 or 128)
        )
        return tiled_mma_SdP, tiled_mma_dKV, tiled_mma_dQ_wg0, tiled_mma_dQ_wg1, tiled_mma_dQ_wg1_192

    def _get_shared_storage_cls(self):
        sQ_alignment = sKV_alignment = sdO_alignment = 1024

        sQ_struct, sKV_struct, sdO_struct = [
            cute.struct.Align[cute.struct.MemRange[type, cute.cosize(layout)], alignment]
            for (layout, type, alignment) in [
                (self.sQ_layout, self.dtype, sQ_alignment),
                (self.sKV_layout, self.dtype, sKV_alignment),
                (self.sdO_layout, self.dtype, sdO_alignment),
            ]
        ]

        cosize_sPdS = cute.cosize(self.sPdS_layout)
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sPdS], 1024]
        sdS_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sPdS], 1024]

        sLSE_struct = cute.struct.Align[cute.struct.MemRange[Float32, cute.round_up(self.tile_m, 64)], 128]
        sdPsum_struct = cute.struct.Align[cute.struct.MemRange[Float32, cute.round_up(self.tile_m, 64)], 128]

        @cute.struct
        class SharedStorage:
            # mbar_QdO for TMA Q/dO sync (replaces pipeline_KV piggyback)
            mbar_QdO: cute.struct.MemRange[cutlass.Int64, 2]
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sQ: sQ_struct
            sKV: sKV_struct
            sdO: sdO_struct
            sP: sP_struct
            sdS: sdS_struct

        return SharedStorage

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mKV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQ: cute.Tensor,
        mdKV: cute.Tensor,
        mTopkIdxs: cute.Tensor,  # (batch, seqlen_q, topk_max) int32
        mTopkLength: cute.Tensor,  # (batch, seqlen_q) int32, per-q valid KV count
        softmax_scale: Float32,
        stream: cuda.CUstream,
    ):
        self._check_type(*(t.element_type for t in (mQ, mKV, mdO, mLSE, mdPsum, mdQ, mdKV)))

        # Assume all strides are divisible by 128 bits except the last stride.
        # Skip cute.assume() for Python ints (e.g. stride=0 from broadcast dims).
        def _assume_strides(t):
            divby = 128 // t.element_type.width
            new_strides = []
            for s in t.stride[:-1]:
                if const_expr(isinstance(s, int)):
                    new_strides.append(s)
                else:
                    new_strides.append(cute.assume(s, divby=divby))
            new_strides.append(t.stride[-1])
            return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=tuple(new_strides)))

        mQ = _assume_strides(mQ)
        mKV = _assume_strides(mKV)
        mdO = _assume_strides(mdO)
        mLSE = _assume_strides(mLSE)
        mdPsum = _assume_strides(mdPsum)
        mdQ = _assume_strides(mdQ)
        mdKV = _assume_strides(mdKV)

        layout_transpose = [1, 3, 2, 0]  # (b, s, n, h) --> (s, h, n, b)
        mQ, mKV, mdO, mdQ = [select(t, layout_transpose) for t in (mQ, mKV, mdO, mdQ)]
        accum_transpose = [2, 1, 0]  # (b, n_kv, s*h) -> (s*h, n_kv, b)
        mdKV = select(mdKV, accum_transpose)

        # Reshape tensor layouts so q-heads within a KV head are packed into M.
        qhpkv = self.qhead_per_kvhead
        num_head_kv = mKV.shape[2]
        mQ, mdO, mdQ = [
            cute.make_tensor(
                mT.iterator,
                cute.make_layout(
                    ((qhpkv, mT.shape[0]), mT.shape[1], num_head_kv, *mT.shape[3:]),
                    stride=((mT.stride[2], mT.stride[0]), mT.stride[1], mT.stride[2] * qhpkv, *mT.stride[3:]),
                ),
            )
            for mT in (mQ, mdO, mdQ)
        ]
        mLSE, mdPsum = [
            cute.make_tensor(
                mT.iterator,
                cute.make_layout(
                    ((qhpkv, mT.shape[1]), num_head_kv, mT.shape[0]),
                    stride=((mT.stride[2], mT.stride[1]), mT.stride[2] * qhpkv, mT.stride[0]),
                ),
            )
            for mT in (mLSE, mdPsum)
        ]

        tiled_mma_SdP, tiled_mma_dKV, tiled_mma_dQ_wg0, tiled_mma_dQ_wg1, tiled_mma_dQ_wg1_192 = self._get_tiled_mma()

        self.num_mma_threads = 128 * self.num_mma_warp_groups
        assert self.num_mma_threads == self.num_threads
        self.num_threads_per_warp_group = 128
        # reg allocation for wg0 and wg1
        self.num_mma_regs = 256

        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }

        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            self.sQ_layout,
            (self.tile_m, self.tile_hdim),
        )
        tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdO,
            self.sdO_layout,
            (self.tile_m, self.tile_hdimv),
        )
        tma_atom_dQ, tma_tensor_dQ = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mdQ,
            self.sQ_half_layout,
            (self.tile_m, 256),
        )
        tma_atom_dQ_64, tma_tensor_dQ_64 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mdQ,
            self.sQ_64_layout,
            (self.tile_m, 64),
        )

        # TileScheduler by Q dimension (m_blocks)
        TileScheduler = SingleTileScheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
            1,  # num_splits
            cute.size(mKV.shape[0]),
            mQ.shape[1],
            mKV.shape[1],
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            mCuSeqlensQ=None,
            mSeqUsedQ=None,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead,
            element_size=self.dtype.width // 8,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E

        qhead_per_kvhead_divmod = FastDivmodDivisor(self.qhead_per_kvhead)

        self.kernel(
            tma_tensor_Q,
            mKV,
            tma_tensor_dO,  # mKV is regular tensor (not TMA)
            tma_tensor_dQ,
            tma_tensor_dQ_64,
            mdKV,
            mTopkIdxs,
            mTopkLength,  # sparse KV indices + per-q length
            tma_atom_Q,
            tma_atom_dO,
            tma_atom_dQ,
            tma_atom_dQ_64,
            mLSE,
            mdPsum,
            self.sQ_layout,
            self.sKV_layout,
            self.sPdS_layout,
            self.sdO_layout,
            self.sKV_quarter_layout,
            self.sKV_192_layout,
            self.sV_layout,
            self.sQ_half_layout,
            self.sQ_64_layout,
            self.sQ_quarter_layout,
            self.sdO_chunk_layout,
            self.sQ_chunk_layout,
            self.r2s_tiled_copy_dKVatomic,
            tiled_mma_SdP,
            tiled_mma_dKV,
            tiled_mma_dQ_wg0,
            tiled_mma_dQ_wg1,
            tiled_mma_dQ_wg1_192,
            softmax_scale_log2,
            softmax_scale,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            qhead_per_kvhead_divmod,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mKV: cute.Tensor,
        mdO: cute.Tensor,
        mdQ: cute.Tensor,
        mdQ_64: cute.Tensor,
        mdKV: cute.Tensor,
        mTopkIdxs: cute.Tensor,  # (batch, seqlen_q, topK_max) int32
        mTopkLength: cute.Tensor,  # (batch, seqlen_q) int32, per-q valid KV count
        tma_atom_Q: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dQ: cute.CopyAtom,
        tma_atom_dQ_64: cute.CopyAtom,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sKV_quarter_layout: cute.ComposedLayout,
        sKV_192_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sQ_half_layout: cute.ComposedLayout,
        sQ_64_layout: cute.ComposedLayout,
        sQ_quarter_layout: cute.ComposedLayout,
        sdO_chunk_layout: cute.ComposedLayout,
        sQ_chunk_layout: cute.ComposedLayout,
        r2s_tiled_copy_dKVatomic: cute.TiledCopy,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dKV: cute.TiledMma,
        tiled_mma_dQ_wg0: cute.TiledMma,
        tiled_mma_dQ_wg1: cute.TiledMma,
        tiled_mma_dQ_wg1_192: cute.TiledMma,
        softmax_scale_log2,
        softmax_scale,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        qhead_per_kvhead_divmod: FastDivmodDivisor,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_dO)
            cpasync.prefetch_descriptor(tma_atom_dQ)
            cpasync.prefetch_descriptor(tma_atom_dQ_64)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        mbar_QdO_ptr = storage.mbar_QdO.data_ptr()
        if warp_idx == 0:
            cute.arch.mbarrier_init(mbar_QdO_ptr, 1)
        cute.arch.sync_threads()

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sKV = storage.sKV.get_tensor(sKV_layout.outer, swizzle=sKV_layout.inner)
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        sP = storage.sP.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sLSE = storage.sLSE.get_tensor(cute.make_layout((self.tile_m,), stride=(1,)))
        sdPsum = storage.sdPsum.get_tensor(cute.make_layout((self.tile_m,), stride=(1,)))

        # sKV subviews for GEMM4 B operand:
        #   d=512: 128 + 128 + 128 + 128 = 512  (WG0: q0+q1, WG1: q2+q3)
        #   d=576: 128 + 128 + 128 + 192 = 576  (WG0: q0+q1, WG1: q2+q3_192)
        sKV_q0 = storage.sKV.get_tensor(sKV_quarter_layout.outer, swizzle=sKV_quarter_layout.inner)
        q_elems_128 = self.tile_n * self.hdim_chunk_dq
        sKV_q1 = cute.make_tensor(sKV_q0.iterator + 1 * q_elems_128, sKV_quarter_layout.outer)
        sKV_q2 = cute.make_tensor(sKV_q0.iterator + 2 * q_elems_128, sKV_quarter_layout.outer)
        sKV_q3_192 = cute.make_tensor(sKV_q0.iterator + 3 * q_elems_128, sKV_192_layout.outer)

        # sV subview: sKV[0:tile_hdimv] = K[0:512] for GEMM2 (dP = dO @ V^T)
        sV = storage.sKV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)

        # WG0 always writes dQ[0:256], WG1 writes dQ[256:tile_hdim]
        # sQ half views for epilogue TMA S2G (both WGs use 256-col halves)
        sQ_half_first = storage.sQ.get_tensor(sQ_half_layout.outer, swizzle=sQ_half_layout.inner)
        sQ_half_second = cute.make_tensor(sQ_half_first.iterator + self.tile_m * 256, sQ_half_layout.outer)
        # sQ_64_epi: cols [512:576] for WG1 epilogue tail TMA (only used when d=576)
        sQ_64_epi = cute.make_tensor(sQ_half_first.iterator + self.tile_m * 512, sQ_64_layout.outer)

        # sQ quarter views for epilogue R2S
        sQ_q0 = storage.sQ.get_tensor(sQ_quarter_layout.outer, swizzle=sQ_quarter_layout.inner)
        sQ_q1 = cute.make_tensor(sQ_q0.iterator + self.tile_m * 128, sQ_quarter_layout.outer)
        sQ_q2 = cute.make_tensor(sQ_q0.iterator + self.tile_m * 256, sQ_quarter_layout.outer)
        sQ_q3 = cute.make_tensor(sQ_q0.iterator + self.tile_m * 384, sQ_quarter_layout.outer)
        # 192-col SMEM view for WG1 epilogue R2S of acc_dQ_3 (cols [384:576])
        sQ_192_epi = cute.make_tensor(sQ_q0.iterator + self.tile_m * 384, sKV_192_layout.outer)

        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        tidx, _, _ = cute.arch.thread_idx()
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)

        if warp_group_idx == 0:
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            self.mma_wg0(
                tiled_mma_SdP,
                tiled_mma_dQ_wg0,
                mQ,
                mKV,
                mdO,
                mLSE,
                mdPsum,
                mTopkIdxs,
                mTopkLength,
                mdQ,
                sQ,
                sKV,
                sV,
                sdO,
                sP,
                sdS,
                sLSE,
                sdPsum,
                sKV_q0,
                sKV_q1,  # GEMM4 B: 128 + 128 (same as V3.1)
                sQ_half_first,
                sQ_q0,
                sQ_q1,  # epilogue: TMA half + R2S quarters
                tma_atom_Q,
                tma_atom_dO,
                tma_atom_dQ,
                mbar_QdO_ptr,
                tidx,
                softmax_scale_log2,
                softmax_scale,
                TileSchedulerCls,
                qhead_per_kvhead_divmod,
            )
        else:
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            self.mma_wg1(
                tiled_mma_dKV,
                tiled_mma_dQ_wg1,
                tiled_mma_dQ_wg1_192,
                mdQ,
                mdQ_64,
                mdKV,
                sQ,
                sKV,
                sdO,
                sP,
                sdS,
                sKV_q2,
                sKV_q3_192,  # GEMM4 B: 128 + 192 (or 128+128 for d=512)
                sQ_half_second,
                sQ_64_epi,
                sQ_192_epi,  # epilogue SMEM views
                sQ_q2,
                sQ_q3,  # epilogue R2S: 128-col quarters
                sQ_64_layout,
                sdO_chunk_layout,
                sQ_chunk_layout,
                r2s_tiled_copy_dKVatomic,
                tma_atom_dQ,
                tma_atom_dQ_64,
                mTopkLength,
                mTopkIdxs,  # per-q length + scatter indices
                tidx,
                TileSchedulerCls,
                qhead_per_kvhead_divmod,
            )

    @cute.jit
    def mma_wg0(
        self,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dQ_wg0: cute.TiledMma,
        mQ: cute.Tensor,
        mKV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mTopkIdxs: cute.Tensor,  # (batch, seqlen_q, topk_max) int32
        mTopkLength: cute.Tensor,  # (batch, seqlen_q) int32, per-q valid KV count
        mdQ: cute.Tensor,
        sQ: cute.Tensor,
        sKV: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sP: cute.Tensor,
        sdS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sKV_q0: cute.Tensor,  # GEMM4 B operand: sKV[:, 0:128]
        sKV_q1: cute.Tensor,  # GEMM4 B operand: sKV[:, 128:256]
        sQ_half: cute.Tensor,  # epilogue TMA S2G: (tile_m, 256)
        sQ_q0: cute.Tensor,  # epilogue R2S: sQ[:, 0:128]
        sQ_q1: cute.Tensor,  # epilogue R2S: sQ[:, 128:256]
        tma_atom_Q: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dQ: cute.CopyAtom,
        mbar_QdO_ptr,
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        TileSchedulerCls: Callable,
        qhead_per_kvhead_divmod: FastDivmodDivisor,
    ):
        """WG0 mainloop: identical to V3.1 — dQ[0:256] = 128+128, no 576-specific logic."""
        wg_tidx = tidx % self.num_threads_per_warp_group  # 0..127

        # Partition fragments for SdP GEMMs (WG0 uses full tidx range 0..127)
        wg_mma_SdP = tiled_mma_SdP.get_slice(wg_tidx)
        thr_mma_SdP = tiled_mma_SdP.get_slice(wg_tidx)

        # GEMM1: S = Q @ KV^T (SS), K dim = tile_hdim (512 or 576)
        tSrQ, tSrKV = mma_partition_fragment_AB(wg_mma_SdP, sQ, sKV, self.SdP_swapAB)
        # GEMM2: dP = dO @ V^T (SS), K dim = tile_hdimv (512)
        tdPrdO, tdPrV = mma_partition_fragment_AB(wg_mma_SdP, sdO, sV, self.SdP_swapAB)

        # GEMM4_WG0: 2 quarter GEMMs — dQ[0:128] and dQ[128:256] (RS, both 128-col)
        wg_mma_dQ = tiled_mma_dQ_wg0.get_slice(wg_tidx)
        sKVt_q0 = transpose_view(sKV_q0)
        sKVt_q1 = transpose_view(sKV_q1)
        _, tdQrKVt_q0 = mma_partition_fragment_AB(wg_mma_dQ, None, sKVt_q0, self.dQ_swapAB)
        _, tdQrKVt_q1 = mma_partition_fragment_AB(wg_mma_dQ, None, sKVt_q1, self.dQ_swapAB)

        # P/dS R2S copy atom
        smem_copy_atom_PdS = get_smem_store_atom(self.arch, self.dtype, transpose=self.SdP_swapAB)
        smem_thr_copy_PdS = cute.make_tiled_copy_C(smem_copy_atom_PdS, tiled_mma_SdP).get_slice(wg_tidx)
        tPsP = smem_thr_copy_PdS.partition_D(sP if const_expr(not self.SdP_swapAB) else transpose_view(sP))
        tdSsdS = smem_thr_copy_PdS.partition_D(sdS if const_expr(not self.SdP_swapAB) else transpose_view(sdS))

        # LSE/dPsum: loaded to SMEM by all 128 threads, then S2R via partition_C
        sLSE_mma = cute.make_tensor(
            sLSE.iterator,
            cute.make_layout((self.tile_m, self.tile_n), stride=(1, 0)),
        )
        sdPsum_mma = cute.make_tensor(
            sdPsum.iterator,
            cute.make_layout((self.tile_m, self.tile_n), stride=(1, 0)),
        )
        if const_expr(self.SdP_swapAB):
            sLSE_mma = transpose_view(sLSE_mma)
            sdPsum_mma = transpose_view(sdPsum_mma)
        LSEslice = (None, 0) if const_expr(not self.SdP_swapAB) else (0, None)
        tLSEsLSE = make_acc_tensor_mn_view(thr_mma_SdP.partition_C(sLSE_mma))[LSEslice]
        tLSEsdPsum = make_acc_tensor_mn_view(thr_mma_SdP.partition_C(sdPsum_mma))[LSEslice]

        # acc_dQ for WG0 — G4_half_0: (64,128), G4_half_1: (64,192)
        dQ_shape_quarter = (self.tile_m, self.hdim_chunk_dq)
        acc_dQ_0 = cute.make_rmem_tensor(
            tiled_mma_dQ_wg0.partition_shape_C(dQ_shape_quarter),
            Float32,
        )
        acc_dQ_1 = cute.make_rmem_tensor(
            tiled_mma_dQ_wg0.partition_shape_C(dQ_shape_quarter),
            Float32,
        )

        mma_qkv_fn = partial(
            gemm_zero_init,
            tiled_mma_SdP,
            (self.tile_m, self.tile_n),
            tSrQ,
            tSrKV,
            swap_AB=self.SdP_swapAB,
        )
        mma_dov_fn = partial(
            gemm_zero_init,
            tiled_mma_SdP,
            (self.tile_m, self.tile_n),
            tdPrdO,
            tdPrV,
            swap_AB=self.SdP_swapAB,
        )
        mma_dsk_fn_0 = partial(
            gemm_w_idx,
            tiled_mma_dQ_wg0,
            acc_dQ_0,
            tCrB=tdQrKVt_q0,
        )
        mma_dsk_fn_1 = partial(
            gemm_w_idx,
            tiled_mma_dQ_wg0,
            acc_dQ_1,
            tCrB=tdQrKVt_q1,
        )

        # cp.async for KV gather
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        # cp.async copy atom: 128-bit = 16B = 8 bf16, bypass L1
        async_copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=128,
        )
        async_thr_copy = cute.make_tiled_copy_tv(
            async_copy_atom,
            cute.make_layout((1,)),
            cute.make_layout((8,)),
        ).get_slice(0)

        # thread organization for cp.async gather
        GROUP_SIZE = const_expr(8)
        idx_in_group = wg_tidx % GROUP_SIZE  # 0..7 dim dir
        group_idx = wg_tidx // GROUP_SIZE  # 0..15 token dir

        mbar_QdO_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            head_idx_kv = head_idx

            # m_block indexes the packed M dimension (qhpkv * seqlen_q / tile_m).
            # topk tensors are indexed by sequence position, not packed m_block.
            # seq_idx = (m_block * tile_m) // qhead_per_kvhead
            seq_idx = (m_block * self.tile_m) // qhead_per_kvhead_divmod

            if const_expr(self.have_topk_length):
                topK = mTopkLength[batch_idx, seq_idx]
            else:
                topK = self.max_topk
            n_block_max = (topK + self.tile_n - 1) // self.tile_n
            topk_tail_rows = topK - (n_block_max - 1) * self.tile_n
            n_block = n_block_max - 1

            mKV_cur = mKV[None, None, head_idx_kv, batch_idx]
            mTopkIdxs_cur = mTopkIdxs[batch_idx, seq_idx, None]

            mQ_cur = mQ[None, None, head_idx, batch_idx]
            gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
            mdO_cur = mdO[None, None, head_idx, batch_idx]
            gdO = cute.local_tile(mdO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))

            load_Q, _, _ = tma_get_copy_fn(tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True)
            load_dO, _, _ = tma_get_copy_fn(tma_atom_dO, 0, cute.make_layout(1), gdO, sdO, single_stage=True)

            if warp_idx_in_wg == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_QdO_ptr, self.tma_copy_bytes["Q"] + self.tma_copy_bytes["dO"])
                load_Q(tma_bar_ptr=mbar_QdO_ptr)
                load_dO(tma_bar_ptr=mbar_QdO_ptr)

            # All 128 threads: load packed-M LSE/dPsum from GMEM to SMEM.
            mLSE_cur = mLSE[None, head_idx, batch_idx]
            mdPsum_cur = mdPsum[None, head_idx, batch_idx]
            num_head = self.qhead_per_kvhead * mLSE.shape[1]
            _load_f32_packed_mh_to_smem(
                mLSE_cur.iterator.toint(),
                sLSE,
                m_block,
                self.tile_m,
                wg_tidx,
                self.num_threads_per_warp_group,
                self.qhead_per_kvhead,
                num_head,
            )
            _load_f32_packed_mh_to_smem(
                mdPsum_cur.iterator.toint(),
                sdPsum,
                m_block,
                self.tile_m,
                wg_tidx,
                self.num_threads_per_warp_group,
                self.qhead_per_kvhead,
                num_head,
            )

            # Fence SMEM writes (LSE/dPsum stores) + barrier to ensure visibility
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.WG0_producer_sync),
                number_of_threads=self.num_threads_per_warp_group,
            )

            # Wait for Q/dO TMA complete
            cute.arch.mbarrier_wait(mbar_QdO_ptr, mbar_QdO_phase)
            mbar_QdO_phase = mbar_QdO_phase ^ 1

            # S2R: LSE and dPsum (SMEM to registers, once per m_block)
            tLSErLSE = load_s2r(tLSEsLSE)
            tLSErdPsum = load_s2r(tLSEsdPsum)

            # first n_block
            self._wg0_one_n_block(
                n_block,
                wg_tidx,
                mKV_cur,
                mTopkIdxs_cur,
                sKV,
                async_copy_atom,
                async_thr_copy,
                idx_in_group,
                group_idx,
                mma_qkv_fn,
                mma_dov_fn,
                mma_dsk_fn_0,
                mma_dsk_fn_1,
                tLSErLSE,
                tLSErdPsum,
                tPsP,
                tdSsdS,
                smem_thr_copy_PdS,
                softmax_scale_log2,
                softmax_scale,
                dQ_accumulate=False,
                is_first=True,
                num_valid_rows=topk_tail_rows,
            )
            n_block -= 1

            # remaining n_blocks
            while n_block >= 0:
                self._wg0_one_n_block(
                    n_block,
                    wg_tidx,
                    mKV_cur,
                    mTopkIdxs_cur,
                    sKV,
                    async_copy_atom,
                    async_thr_copy,
                    idx_in_group,
                    group_idx,
                    mma_qkv_fn,
                    mma_dov_fn,
                    mma_dsk_fn_0,
                    mma_dsk_fn_1,
                    tLSErLSE,
                    tLSErdPsum,
                    tPsP,
                    tdSsdS,
                    smem_thr_copy_PdS,
                    softmax_scale_log2,
                    softmax_scale,
                    dQ_accumulate=True,
                    is_first=False,
                    num_valid_rows=self.tile_n,
                )
                n_block -= 1

            # Wait for WG1 to finish reading sQ (GEMM5 dS^T @ Q uses sQ),
            # before epilogue_dQ overwrites sQ with dQ output.
            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.sdS_consumed),
                number_of_threads=self.num_mma_threads,
            )

            # WG0 always writes dQ[0:256] — same as V3.1 for both d=512 and d=576
            self.epilogue_dQ(
                acc_dQ_0,
                acc_dQ_1,
                mdQ,
                sQ_half,
                sQ_q0,
                sQ_q1,
                tma_atom_dQ,
                tiled_mma_dQ_wg0,
                wg_tidx,
                0,
                m_block,
                head_idx,
                batch_idx,
            )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    # cp.async one gmem row → swizzled smem
    @cute.jit
    def _copy_row(
        self,
        mKV_cur: cute.Tensor,  # (s_kv, headdim) gmem
        mTopkIdxs_cur: cute.Tensor,  # (topk,) gmem
        sKV: cute.Tensor,  # (tile_n, headdim) swizzled smem
        row: Int32,  # smem row index
        idx_in_group: Int32,  # 0..7 dim dir
        copy_atom: cute.CopyAtom,
        thr_copy: cute.TiledCopy,
        global_topk_row: Int32,  # index into topk_idxs
    ):
        token_idx = mTopkIdxs_cur[global_topk_row]
        gKV_row = mKV_cur[token_idx, None]
        gKV_chunks = cute.flat_divide(gKV_row, (8,))
        for tile in cutlass.range_constexpr(self.tile_hdim // 64):
            chunk_idx = tile * 8 + idx_in_group
            g_chunk = gKV_chunks[None, chunk_idx]
            sKV_row = sKV[row, None]
            sKV_row_chunks = cute.flat_divide(sKV_row, (8,))
            s_chunk = sKV_row_chunks[None, chunk_idx]
            tSg = thr_copy.partition_S(g_chunk)
            tSs = thr_copy.partition_D(s_chunk)
            cute.copy(copy_atom, tSg, tSs)

    # clear for OOB rows
    @cute.jit
    def _zero_row(self, sKV: cute.Tensor, row: Int32, idx_in_group: Int32):
        """Zero-fill one K row in smem, cooperative across 8 threads in a group."""
        sK_row = sKV[row, None]
        sK_chunks = cute.flat_divide(sK_row, (8,))
        for tile in cutlass.range_constexpr(self.tile_hdim // 64):
            chunk_idx = tile * 8 + idx_in_group
            sK_chunks[None, chunk_idx].fill(0)

    # Scatter AtomicAdd — write dKV accumulator fragment to
    # per-row interleaved fake-col layout in gmem indexed by topK_idx,
    # using float4 atomics with coalesced access pattern.
    #
    # Interleaved layout per row (hdim_chunk=64):
    #   Per N-tile group (4 ranks × 4 values = 16 f32):
    #     [rank0: 4 vals][rank1: 4 vals][rank2: 4 vals][rank3: 4 vals]
    #   Repeated for 4 N-tiles → 64 values per row.
    #
    # This ensures 4 threads sharing a row write to consecutive 16-byte
    # blocks, achieving coalesced 64-byte access per float4 iteration.
    @cute.jit
    def scatter_dkv_atomic(
        self,
        acc: cute.Tensor,  # MMA accumulator fragment (register, f32)
        chunk_idx: cutlass.Constexpr[int],  # which hdim chunk (0..N_hdim_chunks-1)
        mTopkIdxs_cur: cute.Tensor,  # (topk,) int32 gmem
        mdKVaccum_cur: cute.Tensor,  # (seqlen_k_rounded * hdim_rounded,) f32 gmem
        n_block: Int32,  # current n_block (KV tile index in topk)
        topK: Int32,  # per-q valid KV count (runtime)
        thr_mma: cute.TiledMma,  # thread's MMA slice
        tidx: Int32,
    ):
        """Scatter acc to dKVAccum rows via topK_idx with coalesced float4 atomics.

        Interleaved fake-col addressing:
          fake_pos = (c4 * 4_ranks + rank) * 4 + (c % 4)
                   = c4 * 16 + rank * 4 + (c % 4)
        where c4 = c // 4 (N-tile index), rank = tidx % 4.
        """
        tile_shape = (self.tile_n, self.hdim_chunk)
        cDKV = cute.make_identity_tensor(tile_shape if const_expr(not self.dKV_swapAB) else tile_shape[::-1])
        tScDKV_mn = make_acc_tensor_mn_view(thr_mma.partition_C(cDKV), transpose=self.dKV_swapAB)
        ROW = 0 if const_expr(not self.dKV_swapAB) else 1

        acc_mn = make_acc_tensor_mn_view(acc, transpose=self.dKV_swapAB)
        nrow = const_expr(cute.size(acc_mn.shape[0]))
        ncol = const_expr(cute.size(acc_mn.shape[1]))

        rank = tidx % 4

        for r in cutlass.range_constexpr(nrow):
            local_row = tScDKV_mn[r, 0][ROW]
            global_topk_row = n_block * self.tile_n + local_row
            if global_topk_row < topK:
                global_kv_row = mTopkIdxs_cur[global_topk_row]
                if const_expr(self.have_topk_length) or global_kv_row >= 0:
                    row_base = global_kv_row * self.tile_hdim + chunk_idx * self.hdim_chunk

                    # Each group of 4 MN-view cols maps to one N-tile.
                    # Interleaved: fake_pos = c4 * 16 + rank * 4
                    for c4 in cutlass.range_constexpr(ncol // 4):
                        fake_offset = const_expr(c4) * 16 + rank * 4
                        c = const_expr(c4 * 4)
                        target_ptr = mdKVaccum_cur.iterator + row_base + fake_offset
                        atomic_add_fp32x4(
                            acc_mn[r, c + 0],
                            acc_mn[r, c + 1],
                            acc_mn[r, c + 2],
                            acc_mn[r, c + 3],
                            target_ptr,
                        )

    @cute.jit
    def _wg0_one_n_block(
        self,
        n_block: Int32,
        wg_tidx: Int32,
        mKV_cur: cute.Tensor,  # (s_kv, headdim) gmem
        mTopkIdxs_cur: cute.Tensor,  # (topk,) gmem
        sKV: cute.Tensor,  # (tile_n, headdim) swizzled smem
        async_copy_atom: cute.CopyAtom,
        async_thr_copy: cute.TiledCopy,
        idx_in_group: Int32,
        group_idx: Int32,
        mma_qkv_fn: Callable,
        mma_dov_fn: Callable,
        mma_dsk_fn_0: Callable,  # G4_half_0: dQ[0:128] RS GEMM
        mma_dsk_fn_1: Callable,  # G4_half_1: dQ[128:256] RS GEMM
        tLSErLSE: cute.Tensor,
        tLSErdPsum: cute.Tensor,
        tPsP: cute.Tensor,
        tdSsdS: cute.Tensor,
        smem_thr_copy_PdS: cute.TiledCopy,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        dQ_accumulate: Boolean = False,
        is_first: Boolean = True,
        num_valid_rows: Int32 = 64,
    ):
        """WG0 one n_block: load KV(cp.async) + GEMM1/2 + softmax/dsoftmax + R2S + GEMM4_WG0(x2)"""
        if not is_first:
            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.KV_empty),
                number_of_threads=self.num_mma_threads,  # 256 = WG0(128) + WG1(128)
            )

        # cp.async scatter-gather KV → sKV (all 128 WG0 threads)
        NUM_GROUPS = const_expr(self.num_threads_per_warp_group // 8)  # 16
        ROWS_PER_GROUP = const_expr(self.tile_n // NUM_GROUPS)  # 4
        for r in cutlass.range_constexpr(ROWS_PER_GROUP):
            row = r * NUM_GROUPS + group_idx
            global_topk_row = n_block * self.tile_n + row
            if row < num_valid_rows or not is_first:
                if const_expr(self.have_topk_length):
                    self._copy_row(
                        mKV_cur,
                        mTopkIdxs_cur,
                        sKV,
                        row,
                        idx_in_group,
                        async_copy_atom,
                        async_thr_copy,
                        global_topk_row,
                    )
                else:
                    token_idx = mTopkIdxs_cur[global_topk_row]
                    if token_idx >= 0:
                        self._copy_row(
                            mKV_cur,
                            mTopkIdxs_cur,
                            sKV,
                            row,
                            idx_in_group,
                            async_copy_atom,
                            async_thr_copy,
                            global_topk_row,
                        )
                    else:
                        self._zero_row(sKV, row, idx_in_group)
            else:
                # clear for OOB rows (only in first n_block)
                self._zero_row(sKV, row, idx_in_group)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(
            barrier_id=int(NamedBarrierBwd.WG0_producer_sync),
            number_of_threads=self.num_threads_per_warp_group,
        )

        smem_idx_PdS = 0

        # (1) GEMM1: S = Q @ KV^T
        acc_S = mma_qkv_fn(B_idx=None, wg_wait=-1)

        # (2) GEMM2: dP = dO @ KV^T
        acc_dP = mma_dov_fn(B_idx=None, wg_wait=1)

        # (3) Softmax: P = exp2(S * scale_log2 - LSE)
        acc_S_mn = make_acc_tensor_mn_view(acc_S, transpose=self.SdP_swapAB)
        for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
            for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                acc_S_mn[r, c] = cute.math.exp2(acc_S_mn[r, c] * softmax_scale_log2 - tLSErLSE[r], fastmath=True)

        # Convert P f32 -> bf16
        tdKVrP = cvt_f16(make_acc_tensor_frgA_view(acc_S), self.dtype)

        # R2S P -> sP, then arrive(sP_ready) to notify WG1
        # Wait for sP consumed from previous iteration (PdS_stage==1)
        if dQ_accumulate:
            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.sP_consumed),
                number_of_threads=self.num_mma_threads,
            )
        tPrP = smem_thr_copy_PdS.retile(tdKVrP)
        cute.copy(smem_thr_copy_PdS, tPrP, tPsP[None, None, None, smem_idx_PdS])
        cute.arch.fence_view_async_shared()
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierBwd.sP_ready),
            number_of_threads=self.num_mma_threads,
        )

        # (5) dSoftmax: dS = P * (dP - dPsum)
        warpgroup.wait_group(0)  # GEMM2 done
        acc_dP_mn = make_acc_tensor_mn_view(acc_dP, transpose=self.SdP_swapAB)
        for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
            for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
                acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - tLSErdPsum[r])

        # Convert dS f32 -> bf16 and pre-scale
        tdKVrdS = cvt_f16(make_acc_tensor_frgA_view(acc_dP), self.dtype)
        tdKVrdS_scaled = cute.make_rmem_tensor_like(tdKVrdS, self.dtype)
        for i in cutlass.range_constexpr(cute.size(tdKVrdS)):
            tdKVrdS_scaled[i] = (tdKVrdS[i].to(Float32) * softmax_scale).to(self.dtype)

        # (6) Prepare register A operand for GEMM4 RS GEMMs
        tdQrdS_scaled = cute.make_tensor(
            tdKVrdS_scaled.iterator,
            convert_layout_acc_frgA(acc_dP.layout),
        )

        # (6.5) G4_half_0: dQ[0:128] += dS_scaled(reg) @ sKVt_q0 (RS, immediate)
        # Fire before STS(dS) — dS_scaled is still in registers, TC reads at issue time.
        mma_dsk_fn_0(tCrA=tdQrdS_scaled, B_idx=None, zero_init=not dQ_accumulate, wg_wait=-1)

        # (7) R2S dS_scaled -> sdS (after G4_half_0 issued, regs still valid)
        # Wait for sdS consumed from previous iteration (PdS_stage==1)
        if dQ_accumulate:
            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.sdS_consumed),
                number_of_threads=self.num_mma_threads,
            )
        tdSrdS = smem_thr_copy_PdS.retile(tdKVrdS_scaled)
        cute.copy(smem_thr_copy_PdS, tdSrdS, tdSsdS[None, None, None, smem_idx_PdS])
        cute.arch.fence_view_async_shared()
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierBwd.sdS_ready),
            number_of_threads=self.num_mma_threads,
        )

        # (8) G4_half_1: dQ[128:320] += dS_scaled(reg) @ sKVt_q1_192 (RS, 192-col)
        # Wait for WG1 to signal TC queue has room (after wg_wait(2) on C0)
        cute.arch.barrier(
            barrier_id=int(NamedBarrierBwd.G4_half_ready),
            number_of_threads=self.num_mma_threads,
        )
        mma_dsk_fn_1(tCrA=tdQrdS_scaled, B_idx=None, zero_init=not dQ_accumulate, wg_wait=-1)

        # (9) Wait G4_half_0 + G4_half_1 done. WG0's KV usage is complete.
        # WG0 does NOT explicitly arrive KV_empty — the next iteration's
        # barrier_sync(KV_empty, 256) at the top serves as WG0's arrive + wait.
        warpgroup.wait_group(0)

    @cute.jit
    def mma_wg1(
        self,
        tiled_mma_dKV: cute.TiledMma,
        tiled_mma_dQ_wg1: cute.TiledMma,
        tiled_mma_dQ_wg1_192: cute.TiledMma,
        mdQ: cute.Tensor,
        mdQ_64: cute.Tensor,
        mdKVaccum: cute.Tensor,
        sQ: cute.Tensor,
        sKV: cute.Tensor,
        sdO: cute.Tensor,
        sP: cute.Tensor,
        sdS: cute.Tensor,
        sKV_q2: cute.Tensor,  # GEMM4 B: sKV[:, 256:384] (128-col)
        sKV_q3_192: cute.Tensor,  # GEMM4 B: sKV[:, 384:576] (192-col for d=576, 128-col for d=512)
        sQ_half: cute.Tensor,  # epilogue TMA: (tile_m, 256) — cols [256:512]
        sQ_64_epi: cute.Tensor,  # epilogue TMA tail: (tile_m, 64) — cols [512:576]
        sQ_192_epi: cute.Tensor,  # epilogue R2S: (tile_m, 192) — cols [384:576]
        sQ_q0: cute.Tensor,  # epilogue R2S: sQ[:, 256:384] (128-col)
        sQ_q1: cute.Tensor,  # epilogue R2S: sQ[:, 384:512] (128-col)
        sQ_64_layout: cute.ComposedLayout,
        sdO_chunk_layout: cute.ComposedLayout,
        sQ_chunk_layout: cute.ComposedLayout,
        r2s_tiled_copy_dKVatomic: cute.TiledCopy,
        tma_atom_dQ: cute.CopyAtom,
        tma_atom_dQ_64: cute.CopyAtom,
        mTopkLength: cute.Tensor,  # (batch, seqlen_q) int32, per-q valid KV count
        mTopkIdxs: cute.Tensor,  # (batch, seqlen_q, topk_max) int32 for scatter
        tidx: Int32,
        TileSchedulerCls: Callable,
        qhead_per_kvhead_divmod: FastDivmodDivisor,
    ):
        """WG1 mainloop: 9-chunk dKV pipeline + 2 G4_half (dQ) + AtomicAdd scatter."""
        wg_tidx = tidx % self.num_threads_per_warp_group  # 0..127

        # --- dKV GEMM A-operand partitions (same for all chunks) ---
        wg_mma_dKV = tiled_mma_dKV.get_slice(wg_tidx)
        # GEMM3 A: P^T (MN-major transpose view)
        sPt = transpose_view(sP)
        tdKVrPt, _ = mma_partition_fragment_AB(wg_mma_dKV, sPt, None, self.dKV_swapAB)
        # GEMM5 A: dS^T (MN-major transpose view)
        sdSt = transpose_view(sdS)
        tdKVrdSt, _ = mma_partition_fragment_AB(wg_mma_dKV, sdSt, None, self.dKV_swapAB)

        # 2-stage acc_dKV — two fragments alternating s0(even)/s1(odd)
        # to overlap AtomicAdd latency with TC computation via wg_wait(2).
        dKV_chunk_shape = (self.tile_n, self.hdim_chunk)
        acc_dKV_part_shape = tiled_mma_dKV.partition_shape_C(dKV_chunk_shape if not self.dKV_swapAB else dKV_chunk_shape[::-1])
        acc_dKV_s0 = cute.make_rmem_tensor(acc_dKV_part_shape, Float32)
        acc_dKV_s1 = cute.make_rmem_tensor(acc_dKV_part_shape, Float32)

        # --- dQ GEMM4_WG1 partitions: G4_half_2 (128-col) + G4_half_3 (192 or 128-col) ---
        wg_mma_dQ = tiled_mma_dQ_wg1.get_slice(wg_tidx)
        wg_mma_dQ_192 = tiled_mma_dQ_wg1_192.get_slice(wg_tidx)
        sKVt_q2 = transpose_view(sKV_q2)
        sKVt_q3_192 = transpose_view(sKV_q3_192)
        tdQrsdS, tdQrKVt_q2 = mma_partition_fragment_AB(wg_mma_dQ, sdS, sKVt_q2, self.dQ_swapAB)
        # G4_half_3: partition with 192-col tiled_mma (or 128-col for d=512, same layout)
        tdQrsdS_192, tdQrKVt_q3_192 = mma_partition_fragment_AB(wg_mma_dQ_192, sdS, sKVt_q3_192, self.dQ_swapAB)

        dQ_shape_quarter = (self.tile_m, self.hdim_chunk_dq)
        acc_dQ_2 = cute.make_rmem_tensor(
            tiled_mma_dQ_wg1.partition_shape_C(dQ_shape_quarter),
            Float32,
        )
        dQ_shape_wg1_1 = (self.tile_m, self.hdim_chunk_dq_wg1_1)
        acc_dQ_3 = cute.make_rmem_tensor(
            tiled_mma_dQ_wg1_192.partition_shape_C(dQ_shape_wg1_1),
            Float32,
        )

        mma_dQ_wg1_fn_2 = partial(
            gemm_w_idx,
            tiled_mma_dQ_wg1,
            acc_dQ_2,
            tdQrsdS,
            tdQrKVt_q2,
            swap_AB=self.dQ_swapAB,
        )
        mma_dQ_wg1_fn_3 = partial(
            gemm_w_idx,
            tiled_mma_dQ_wg1_192,
            acc_dQ_3,
            tdQrsdS_192,
            tdQrKVt_q3_192,
            swap_AB=self.dQ_swapAB,
        )

        # --- Scatter AtomicAdd setup ---
        thr_mma_dKV = tiled_mma_dKV.get_slice(wg_tidx)

        # Base pointers for chunk views (ptr offset pattern)
        chunk_elems = self.tile_m * self.hdim_chunk

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            head_idx_kv = head_idx

            # Convert packed-M m_block to sequence index for topk tensors.
            seq_idx = (m_block * self.tile_m) // qhead_per_kvhead_divmod

            if const_expr(self.have_topk_length):
                topK = mTopkLength[batch_idx, seq_idx]
            else:
                topK = self.max_topk
            n_block_max = (topK + self.tile_n - 1) // self.tile_n

            # scatter AtomicAdd — flat gmem dKVaccum for per-row addressing
            mdKVaccum_cur = mdKVaccum[None, head_idx_kv, batch_idx]
            mTopkIdxs_cur = mTopkIdxs[batch_idx, seq_idx, None]

            n_block = n_block_max - 1
            first_iter = True
            while n_block >= 0:
                smem_idx_PdS = 0  # PdS_stage=1

                # ===== 9-chunk 2-stage dKV pipeline (hdim_chunk=64, 576-dim) =====
                # c0-c7: GEMM3(P^T@dO) + GEMM5(dS^T@Q), dO has 8 chunks (512/64)
                # c8: GEMM5 only (dS^T@Q[512:576]), no GEMM3 (dO only 512 dim)
                # Pipeline: C0(s0) → [sdS_ready] → C0(s0) → C1(s1) → wg_wait(2) →
                # arrive(G4_half_ready) → AtomicAdd(C0) → C2(s0) → wg_wait(2) →
                # G4_half_2 → AtomicAdd(C1) → C3(s1) → wg_wait(2) → G4_half_3 →
                # AtomicAdd(C2) → C4(s0) → wg_wait(2) → KV_empty → AtomicAdd(C3) →
                # C5(s1) → wg_wait(2) → AtomicAdd(C4) → C6(s0) → wg_wait(2) →
                # AtomicAdd(C5) → C7(s1) → wg_wait(2) → AtomicAdd(C6) →
                # C8(s0, GEMM5 only) → wg_wait(0) → sP_consumed + sdS_consumed →
                # AtomicAdd(C7) → AtomicAdd(C8)
                # --- Wait sP_ready ---
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.sP_ready),
                    number_of_threads=self.num_mma_threads,
                )

                # --- c0 (stage 0): dKV_s0 = P^T @ dO[0:64] + dS^T @ Q[0:64] ---
                sdO_c0 = cute.make_tensor(sdO.iterator + 0 * chunk_elems, sdO_chunk_layout.outer)
                sQ_c0 = cute.make_tensor(sQ.iterator + 0 * chunk_elems, sQ_chunk_layout.outer)
                _, tdKVrdOt_c0 = mma_partition_fragment_AB(wg_mma_dKV, None, transpose_view(sdO_c0), self.dKV_swapAB)
                _, tdKVrQt_c0 = mma_partition_fragment_AB(wg_mma_dKV, None, transpose_view(sQ_c0), self.dKV_swapAB)

                # GEMM3_c0: sP already ready
                gemm_w_idx(tiled_mma_dKV, acc_dKV_s0, tdKVrPt, tdKVrdOt_c0, A_idx=smem_idx_PdS, zero_init=True, wg_wait=-1, swap_AB=self.dKV_swapAB)

                # GEMM5_c0: wait sdS_ready
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.sdS_ready),
                    number_of_threads=self.num_mma_threads,
                )
                gemm_w_idx(tiled_mma_dKV, acc_dKV_s0, tdKVrdSt, tdKVrQt_c0, A_idx=smem_idx_PdS, zero_init=False, wg_wait=-1, swap_AB=self.dKV_swapAB)

                # --- c1 (stage 1): dKV_s1 = P^T @ dO[64:128] + dS^T @ Q[64:128] ---
                sdO_c1 = cute.make_tensor(sdO.iterator + 1 * chunk_elems, sdO_chunk_layout.outer)
                sQ_c1 = cute.make_tensor(sQ.iterator + 1 * chunk_elems, sQ_chunk_layout.outer)
                _, tdKVrdOt_c1 = mma_partition_fragment_AB(wg_mma_dKV, None, transpose_view(sdO_c1), self.dKV_swapAB)
                _, tdKVrQt_c1 = mma_partition_fragment_AB(wg_mma_dKV, None, transpose_view(sQ_c1), self.dKV_swapAB)

                gemm_w_idx(tiled_mma_dKV, acc_dKV_s1, tdKVrPt, tdKVrdOt_c1, A_idx=smem_idx_PdS, zero_init=True, wg_wait=-1, swap_AB=self.dKV_swapAB)
                gemm_w_idx(tiled_mma_dKV, acc_dKV_s1, tdKVrdSt, tdKVrQt_c1, A_idx=smem_idx_PdS, zero_init=False, wg_wait=-1, swap_AB=self.dKV_swapAB)

                # --- wg_wait(2): c0 done (G3_C0 + G5_C0 consumed, C1 pending) ---
                warpgroup.wait_group(2)

                # Signal WG0 that TC queue has room for G4_half_1
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierBwd.G4_half_ready),
                    number_of_threads=self.num_mma_threads,
                )

                # AtomicAdd C0 (s0)
                self.scatter_dkv_atomic(
                    acc_dKV_s0,
                    0,
                    mTopkIdxs_cur,
                    mdKVaccum_cur,
                    n_block,
                    topK,
                    thr_mma_dKV,
                    tidx,
                )

                # --- c2 (stage 0, reuse s0): P^T @ dO[128:192] + dS^T @ Q[128:192] ---
                sdO_c2 = cute.make_tensor(sdO.iterator + 2 * chunk_elems, sdO_chunk_layout.outer)
                sQ_c2 = cute.make_tensor(sQ.iterator + 2 * chunk_elems, sQ_chunk_layout.outer)
                _, tdKVrdOt_c2 = mma_partition_fragment_AB(wg_mma_dKV, None, transpose_view(sdO_c2), self.dKV_swapAB)
                _, tdKVrQt_c2 = mma_partition_fragment_AB(wg_mma_dKV, None, transpose_view(sQ_c2), self.dKV_swapAB)

                gemm_w_idx(tiled_mma_dKV, acc_dKV_s0, tdKVrPt, tdKVrdOt_c2, A_idx=smem_idx_PdS, zero_init=True, wg_wait=-1, swap_AB=self.dKV_swapAB)
                gemm_w_idx(tiled_mma_dKV, acc_dKV_s0, tdKVrdSt, tdKVrQt_c2, A_idx=smem_idx_PdS, zero_init=False, wg_wait=-1, swap_AB=self.dKV_swapAB)

                # --- wg_wait(2): c1 done ---
                warpgroup.wait_group(2)

                # G4_half_2: dQ[320:448] += sdS @ sKVt_q2 (SS, fills TC gap)
                mma_dQ_wg1_fn_2(
                    A_idx=smem_idx_PdS,
                    B_idx=None,
                    zero_init=first_iter,
                    wg_wait=-1,
                )

                # AtomicAdd C1 (s1)
                self.scatter_dkv_atomic(
                    acc_dKV_s1,
                    1,
                    mTopkIdxs_cur,
                    mdKVaccum_cur,
                    n_block,
                    topK,
                    thr_mma_dKV,
                    tidx,
                )

                # --- c3 (stage 1, reuse s1): P^T @ dO[192:256] + dS^T @ Q[192:256] ---
                sdO_c3 = cute.make_tensor(sdO.iterator + 3 * chunk_elems, sdO_chunk_layout.outer)
                sQ_c3 = cute.make_tensor(sQ.iterator + 3 * chunk_elems, sQ_chunk_layout.outer)
                _, tdKVrdOt_c3 = mma_partition_fragment_AB(wg_mma_dKV, None, transpose_view(sdO_c3), self.dKV_swapAB)
                _, tdKVrQt_c3 = mma_partition_fragment_AB(wg_mma_dKV, None, transpose_view(sQ_c3), self.dKV_swapAB)

                gemm_w_idx(tiled_mma_dKV, acc_dKV_s1, tdKVrPt, tdKVrdOt_c3, A_idx=smem_idx_PdS, zero_init=True, wg_wait=-1, swap_AB=self.dKV_swapAB)
                gemm_w_idx(tiled_mma_dKV, acc_dKV_s1, tdKVrdSt, tdKVrQt_c3, A_idx=smem_idx_PdS, zero_init=False, wg_wait=-1, swap_AB=self.dKV_swapAB)

                # --- wg_wait(2): c2 + G4_half_2 done ---
                warpgroup.wait_group(2)

                # G4_half_3: dQ[384:576] += sdS @ sKVt_q3_192 (SS, 192-col for d=576)
                mma_dQ_wg1_fn_3(
                    A_idx=smem_idx_PdS,
                    B_idx=None,
                    zero_init=first_iter,
                    wg_wait=-1,
                )

                # AtomicAdd C2 (s0)
                self.scatter_dkv_atomic(
                    acc_dKV_s0,
                    2,
                    mTopkIdxs_cur,
                    mdKVaccum_cur,
                    n_block,
                    topK,
                    thr_mma_dKV,
                    tidx,
                )

                # --- c4 (stage 0, reuse s0): P^T @ dO[256:320] + dS^T @ Q[256:320] ---
                sdO_c4 = cute.make_tensor(sdO.iterator + 4 * chunk_elems, sdO_chunk_layout.outer)
                sQ_c4 = cute.make_tensor(sQ.iterator + 4 * chunk_elems, sQ_chunk_layout.outer)
                _, tdKVrdOt_c4 = mma_partition_fragment_AB(wg_mma_dKV, None, transpose_view(sdO_c4), self.dKV_swapAB)
                _, tdKVrQt_c4 = mma_partition_fragment_AB(wg_mma_dKV, None, transpose_view(sQ_c4), self.dKV_swapAB)

                gemm_w_idx(tiled_mma_dKV, acc_dKV_s0, tdKVrPt, tdKVrdOt_c4, A_idx=smem_idx_PdS, zero_init=True, wg_wait=-1, swap_AB=self.dKV_swapAB)
                gemm_w_idx(tiled_mma_dKV, acc_dKV_s0, tdKVrdSt, tdKVrQt_c4, A_idx=smem_idx_PdS, zero_init=False, wg_wait=-1, swap_AB=self.dKV_swapAB)

                # --- wg_wait(2): c3 + G4_half_3 done ---
                warpgroup.wait_group(2)

                # WG1 done reading sKV (G4_half_3 was last sKV user) — arrive KV_empty
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierBwd.KV_empty),
                    number_of_threads=self.num_mma_threads,  # 256
                )

                # AtomicAdd C3 (s1)
                self.scatter_dkv_atomic(
                    acc_dKV_s1,
                    3,
                    mTopkIdxs_cur,
                    mdKVaccum_cur,
                    n_block,
                    topK,
                    thr_mma_dKV,
                    tidx,
                )

                # --- c5 (stage 1, reuse s1): P^T @ dO[320:384] + dS^T @ Q[320:384] ---
                sdO_c5 = cute.make_tensor(sdO.iterator + 5 * chunk_elems, sdO_chunk_layout.outer)
                sQ_c5 = cute.make_tensor(sQ.iterator + 5 * chunk_elems, sQ_chunk_layout.outer)
                _, tdKVrdOt_c5 = mma_partition_fragment_AB(wg_mma_dKV, None, transpose_view(sdO_c5), self.dKV_swapAB)
                _, tdKVrQt_c5 = mma_partition_fragment_AB(wg_mma_dKV, None, transpose_view(sQ_c5), self.dKV_swapAB)

                gemm_w_idx(tiled_mma_dKV, acc_dKV_s1, tdKVrPt, tdKVrdOt_c5, A_idx=smem_idx_PdS, zero_init=True, wg_wait=-1, swap_AB=self.dKV_swapAB)
                gemm_w_idx(tiled_mma_dKV, acc_dKV_s1, tdKVrdSt, tdKVrQt_c5, A_idx=smem_idx_PdS, zero_init=False, wg_wait=-1, swap_AB=self.dKV_swapAB)

                # --- wg_wait(2): c4 done ---
                warpgroup.wait_group(2)

                # AtomicAdd C4 (s0)
                self.scatter_dkv_atomic(
                    acc_dKV_s0,
                    4,
                    mTopkIdxs_cur,
                    mdKVaccum_cur,
                    n_block,
                    topK,
                    thr_mma_dKV,
                    tidx,
                )

                # --- c6 (stage 0, reuse s0): P^T @ dO[384:448] + dS^T @ Q[384:448] ---
                sdO_c6 = cute.make_tensor(sdO.iterator + 6 * chunk_elems, sdO_chunk_layout.outer)
                sQ_c6 = cute.make_tensor(sQ.iterator + 6 * chunk_elems, sQ_chunk_layout.outer)
                _, tdKVrdOt_c6 = mma_partition_fragment_AB(wg_mma_dKV, None, transpose_view(sdO_c6), self.dKV_swapAB)
                _, tdKVrQt_c6 = mma_partition_fragment_AB(wg_mma_dKV, None, transpose_view(sQ_c6), self.dKV_swapAB)

                gemm_w_idx(tiled_mma_dKV, acc_dKV_s0, tdKVrPt, tdKVrdOt_c6, A_idx=smem_idx_PdS, zero_init=True, wg_wait=-1, swap_AB=self.dKV_swapAB)
                gemm_w_idx(tiled_mma_dKV, acc_dKV_s0, tdKVrdSt, tdKVrQt_c6, A_idx=smem_idx_PdS, zero_init=False, wg_wait=-1, swap_AB=self.dKV_swapAB)

                # --- wg_wait(2): c5 done ---
                warpgroup.wait_group(2)

                # AtomicAdd C5 (s1)
                self.scatter_dkv_atomic(
                    acc_dKV_s1,
                    5,
                    mTopkIdxs_cur,
                    mdKVaccum_cur,
                    n_block,
                    topK,
                    thr_mma_dKV,
                    tidx,
                )

                # --- c7 (stage 1, reuse s1): P^T @ dO[448:512] + dS^T @ Q[448:512] ---
                sdO_c7 = cute.make_tensor(sdO.iterator + 7 * chunk_elems, sdO_chunk_layout.outer)
                sQ_c7 = cute.make_tensor(sQ.iterator + 7 * chunk_elems, sQ_chunk_layout.outer)
                _, tdKVrdOt_c7 = mma_partition_fragment_AB(wg_mma_dKV, None, transpose_view(sdO_c7), self.dKV_swapAB)
                _, tdKVrQt_c7 = mma_partition_fragment_AB(wg_mma_dKV, None, transpose_view(sQ_c7), self.dKV_swapAB)

                gemm_w_idx(tiled_mma_dKV, acc_dKV_s1, tdKVrPt, tdKVrdOt_c7, A_idx=smem_idx_PdS, zero_init=True, wg_wait=-1, swap_AB=self.dKV_swapAB)
                gemm_w_idx(tiled_mma_dKV, acc_dKV_s1, tdKVrdSt, tdKVrQt_c7, A_idx=smem_idx_PdS, zero_init=False, wg_wait=-1, swap_AB=self.dKV_swapAB)

                # --- wg_wait(2): c6 done ---
                warpgroup.wait_group(2)

                # AtomicAdd C6 (s0)
                self.scatter_dkv_atomic(
                    acc_dKV_s0,
                    6,
                    mTopkIdxs_cur,
                    mdKVaccum_cur,
                    n_block,
                    topK,
                    thr_mma_dKV,
                    tidx,
                )

                if const_expr(not self.same_hdim_kv):
                    # --- d=576: c8 (stage 0, reuse s0): GEMM5 only — dS^T @ Q[512:576] ---
                    sQ_c8 = cute.make_tensor(sQ.iterator + 8 * chunk_elems, sQ_chunk_layout.outer)
                    _, tdKVrQt_c8 = mma_partition_fragment_AB(wg_mma_dKV, None, transpose_view(sQ_c8), self.dKV_swapAB)

                    gemm_w_idx(tiled_mma_dKV, acc_dKV_s0, tdKVrdSt, tdKVrQt_c8, A_idx=smem_idx_PdS, zero_init=True, wg_wait=0, swap_AB=self.dKV_swapAB)

                    # Release sP/sdS
                    cute.arch.barrier_arrive(
                        barrier_id=int(NamedBarrierBwd.sP_consumed),
                        number_of_threads=self.num_mma_threads,
                    )
                    cute.arch.barrier_arrive(
                        barrier_id=int(NamedBarrierBwd.sdS_consumed),
                        number_of_threads=self.num_mma_threads,
                    )

                    # AtomicAdd C7 (s1) + C8 (s0)
                    self.scatter_dkv_atomic(
                        acc_dKV_s1,
                        7,
                        mTopkIdxs_cur,
                        mdKVaccum_cur,
                        n_block,
                        topK,
                        thr_mma_dKV,
                        tidx,
                    )
                    self.scatter_dkv_atomic(
                        acc_dKV_s0,
                        8,
                        mTopkIdxs_cur,
                        mdKVaccum_cur,
                        n_block,
                        topK,
                        thr_mma_dKV,
                        tidx,
                    )
                else:
                    # --- d=512: original c7 drain (no c8) ---
                    warpgroup.wait_group(0)

                    # Release sP/sdS
                    cute.arch.barrier_arrive(
                        barrier_id=int(NamedBarrierBwd.sP_consumed),
                        number_of_threads=self.num_mma_threads,
                    )
                    cute.arch.barrier_arrive(
                        barrier_id=int(NamedBarrierBwd.sdS_consumed),
                        number_of_threads=self.num_mma_threads,
                    )

                    # AtomicAdd C7 (s1)
                    self.scatter_dkv_atomic(
                        acc_dKV_s1,
                        7,
                        mTopkIdxs_cur,
                        mdKVaccum_cur,
                        n_block,
                        topK,
                        thr_mma_dKV,
                        tidx,
                    )

                n_block -= 1
                first_iter = False

            # epilogue: write dQ[256:tile_hdim] to gmem
            # WG1 writes dQ[256:tile_hdim]: 128+128 for d=512, 128+192 for d=576
            self.epilogue_dQ_wg1(
                acc_dQ_2,
                acc_dQ_3,
                mdQ,
                mdQ_64,
                sQ_half,
                sQ_64_epi,
                sQ_q0,
                sQ_192_epi,
                tma_atom_dQ,
                tma_atom_dQ_64,
                tiled_mma_dQ_wg1,
                tiled_mma_dQ_wg1_192,
                wg_tidx,
                m_block,
                head_idx,
                batch_idx,
            )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def epilogue_dQ_wg1(
        self,
        acc_dQ_2: cute.Tensor,  # (tile_m, 128) accumulator
        acc_dQ_3: cute.Tensor,  # (tile_m, hdim_chunk_dq_wg1_1) accumulator (128 or 192)
        mdQ: cute.Tensor,
        mdQ_64: cute.Tensor,
        sQ_half: cute.Tensor,  # (tile_m, 256) for TMA S2G — cols [256:512]
        sQ_64_epi: cute.Tensor,  # (tile_m, 64) for tail TMA — cols [512:576]
        sQ_q2: cute.Tensor,  # (tile_m, 128) R2S target — cols [256:384]
        sQ_192_epi: cute.Tensor,  # (tile_m, wg1_1) R2S target — cols [384:512] or [384:576]
        tma_atom_dQ: cute.CopyAtom,
        tma_atom_dQ_64: cute.CopyAtom,
        tiled_mma_dQ_128: cute.TiledMma,
        tiled_mma_dQ_192: cute.TiledMma,
        wg_tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        """WG1 epilogue: write dQ[256:tile_hdim] to gmem.
        d=512: 128+128 R2S → 256-col TMA
        d=576: 128+192 R2S → 256-col TMA + 64-col tail TMA
        """
        # R2S acc_dQ_2(128) → sQ_q2 [256:384]
        smem_copy_atom_128 = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=self.dQ_swapAB, num_matrices=4),
            self.dtype,
        )
        smem_thr_copy_128 = cute.make_tiled_copy_C(smem_copy_atom_128, tiled_mma_dQ_128).get_slice(wg_tidx)

        rdQ2 = cute.make_rmem_tensor_like(acc_dQ_2, self.dtype)
        rdQ2.store(acc_dQ_2.load().to(self.dtype))
        taccdQ2rdQ2 = smem_thr_copy_128.retile(rdQ2)
        sdQ2 = sQ_q2 if const_expr(not self.dQ_swapAB) else transpose_view(sQ_q2)
        taccdQ2sdQ2 = smem_thr_copy_128.partition_D(sdQ2)
        cute.copy(smem_copy_atom_128, taccdQ2rdQ2, taccdQ2sdQ2)

        # R2S acc_dQ_3(wg1_1) → sQ_192_epi [384:384+wg1_1]
        smem_copy_atom_192 = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=self.dQ_swapAB, num_matrices=4),
            self.dtype,
        )
        smem_thr_copy_192 = cute.make_tiled_copy_C(smem_copy_atom_192, tiled_mma_dQ_192).get_slice(wg_tidx)

        rdQ3 = cute.make_rmem_tensor_like(acc_dQ_3, self.dtype)
        rdQ3.store(acc_dQ_3.load().to(self.dtype))
        taccdQ3rdQ3 = smem_thr_copy_192.retile(rdQ3)
        sdQ3_192 = sQ_192_epi if const_expr(not self.dQ_swapAB) else transpose_view(sQ_192_epi)
        taccdQ3sdQ3 = smem_thr_copy_192.partition_D(sdQ3_192)
        cute.copy(smem_copy_atom_192, taccdQ3rdQ3, taccdQ3sdQ3)

        # Fence + barrier
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(
            barrier_id=int(NamedBarrierBwd.Epilogue_WG1),
            number_of_threads=self.num_threads_per_warp_group,
        )

        # TMA S2G: sQ_half(256) → gdQ[:, 256:512]
        warp_idx_in_wg = wg_tidx // 32
        if warp_idx_in_wg == 0:
            mdQ_cur = mdQ[None, None, head_idx, batch_idx]
            gdQ_half = cute.local_tile(mdQ_cur, (self.tile_m, 256), (m_block, 1))  # col_block=1 → offset 256
            with cute.arch.elect_one():
                store_dQ, _, _ = tma_get_copy_fn(tma_atom_dQ, 0, cute.make_layout(1), sQ_half, gdQ_half, single_stage=True)
                store_dQ()
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)

        # Tail TMA: sQ_64_epi(64) → gdQ[:, 512:576] (only for d=576)
        if const_expr(not self.same_hdim_kv):
            if warp_idx_in_wg == 0:
                mdQ_64_cur = mdQ_64[None, None, head_idx, batch_idx]
                gdQ_64 = cute.local_tile(mdQ_64_cur, (self.tile_m, 64), (m_block, 8))  # col_block=8 → offset 512
                with cute.arch.elect_one():
                    store_dQ_64, _, _ = tma_get_copy_fn(tma_atom_dQ_64, 0, cute.make_layout(1), sQ_64_epi, gdQ_64, single_stage=True)
                    store_dQ_64()
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def epilogue_dQ(
        self,
        acc_dQ_0: cute.Tensor,
        acc_dQ_1: cute.Tensor,
        mdQ: cute.Tensor,
        sQ_half: cute.Tensor,
        sQ_q0: cute.Tensor,
        sQ_q1: cute.Tensor,
        tma_atom_dQ: cute.CopyAtom,
        tiled_mma_dQ: cute.TiledMma,
        wg_tidx: Int32,
        wg_idx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        """Original epilogue for d=512: 2 quarter R2S → 1 TMA S2G (256-col).
        wg_idx: 0 → dQ[0:256], 1 → dQ[256:512]
        """
        smem_copy_atom_dQ = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=self.dQ_swapAB, num_matrices=4),
            self.dtype,
        )
        smem_thr_copy_dQ = cute.make_tiled_copy_C(smem_copy_atom_dQ, tiled_mma_dQ).get_slice(wg_tidx)

        rdQ0 = cute.make_rmem_tensor_like(acc_dQ_0, self.dtype)
        rdQ0.store(acc_dQ_0.load().to(self.dtype))
        taccdQ0rdQ0 = smem_thr_copy_dQ.retile(rdQ0)
        sdQ0 = sQ_q0 if const_expr(not self.dQ_swapAB) else transpose_view(sQ_q0)
        taccdQ0sdQ0 = smem_thr_copy_dQ.partition_D(sdQ0)
        cute.copy(smem_copy_atom_dQ, taccdQ0rdQ0, taccdQ0sdQ0)

        rdQ1 = cute.make_rmem_tensor_like(acc_dQ_1, self.dtype)
        rdQ1.store(acc_dQ_1.load().to(self.dtype))
        taccdQ1rdQ1 = smem_thr_copy_dQ.retile(rdQ1)
        sdQ1 = sQ_q1 if const_expr(not self.dQ_swapAB) else transpose_view(sQ_q1)
        taccdQ1sdQ1 = smem_thr_copy_dQ.partition_D(sdQ1)
        cute.copy(smem_copy_atom_dQ, taccdQ1rdQ1, taccdQ1sdQ1)

        cute.arch.fence_view_async_shared()
        epilogue_bar_id = int(NamedBarrierBwd.Epilogue_WG0) if wg_idx == 0 else int(NamedBarrierBwd.Epilogue_WG1)
        cute.arch.barrier(
            barrier_id=epilogue_bar_id,
            number_of_threads=self.num_threads_per_warp_group,
        )

        warp_idx_in_wg = wg_tidx // 32
        if warp_idx_in_wg == 0:
            mdQ_cur = mdQ[None, None, head_idx, batch_idx]
            gdQ_half = cute.local_tile(mdQ_cur, (self.tile_m, 256), (m_block, wg_idx))
            with cute.arch.elect_one():
                store_dQ, _, _ = tma_get_copy_fn(tma_atom_dQ, 0, cute.make_layout(1), sQ_half, gdQ_half, single_stage=True)
                store_dQ()
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)


class _FlashAttentionDSABackwardPostprocessSm90:
    """Postprocess kernel: fake-col f32 dKVAccum -> real-col bf16 dKV.

    The backward kernel accumulates dKV gradients via scatter atomicAdd into
    a flat f32 buffer with an interleaved "fake column" layout dictated by the
    SM90 GMMA MN-major accumulator fragment ordering.

    This kernel reads the accumulator, applies the inverse permutation, converts
    f32 -> bf16, transposes axes (batch, nkv, seqlen_k) -> (batch, seqlen_k, nkv),
    and writes the final dKV output.

    Pipeline per CTA (tile_n rows x hdim_chunk cols):
      Phase 1: gmem f32 (fake-col) -> cvt f32->bf16 -> smem (real-col)
      Phase 2: smem (real-col, row-major) -> coalesced gmem bf16 write

    Fake-col interleaving within each 16-element N-tile group:
      fake_col = nt*16 + rank*4 + k       (nt: N-tile idx, rank: 0-3, k: 0-3)
      real_col = nt*16 + rank*2 + (k//2)*8 + (k%2)
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        hdim_chunk: int = 128,
        tile_n: int = 64,
        head_dim: int = 512,
        num_threads: int = 128,
        N_hdim_chunks: int = 4,
    ):
        self.dtype = dtype
        self.hdim_chunk = hdim_chunk
        self.tile_n = tile_n
        self.head_dim = head_dim
        self.num_threads = num_threads
        self.N_hdim_chunks = N_hdim_chunks
        hdim_multiple_of = 32
        self.head_dim_rounded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        assert hdim_chunk % 16 == 0, "hdim_chunk must be a multiple of 16"
        assert num_threads == hdim_chunk, "num_threads must equal hdim_chunk so each thread handles one fake-col"

    @cute.jit
    def __call__(
        self,
        mdKVaccum: cute.Tensor,
        mdKV: cute.Tensor,
        seqlen_k: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        num_head_kv = mdKVaccum.shape[1]
        num_batch = mdKVaccum.shape[0]
        seqlen_k_rounded = mdKVaccum.shape[2] // self.head_dim_rounded
        num_n_blocks = cute.ceil_div(seqlen_k_rounded, self.tile_n)
        num_blocks = num_n_blocks * self.N_hdim_chunks

        smem_bytes = self.tile_n * self.hdim_chunk * (self.dtype.width // 8)

        self.kernel(mdKVaccum, mdKV, seqlen_k).launch(
            grid=[num_blocks, num_head_kv, num_batch],
            block=[self.num_threads, 1, 1],
            smem=smem_bytes,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mdKVaccum: cute.Tensor,
        mdKV: cute.Tensor,
        seqlen_k: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block = cute.arch.block_idx()[0]
        head_idx = cute.arch.block_idx()[1]
        batch_idx = cute.arch.block_idx()[2]

        n_block = m_block // self.N_hdim_chunks
        chunk_k = m_block % self.N_hdim_chunks

        mdKVaccum_cur = mdKVaccum[batch_idx, head_idx, None]
        mdKV_cur = mdKV[batch_idx, None, head_idx, None]

        sdKV_ptr = cute.arch.get_dyn_smem(self.dtype)
        sdKV = cute.make_tensor(
            sdKV_ptr,
            cute.make_layout(
                (self.tile_n, self.hdim_chunk),
                stride=(self.hdim_chunk, 1),
            ),
        )

        self._fake_to_smem(
            mdKVaccum_cur,
            sdKV,
            tidx,
            n_block,
            chunk_k,
            seqlen_k,
        )
        cute.arch.sync_threads()
        self._smem_to_gmem(
            sdKV,
            mdKV_cur,
            tidx,
            n_block,
            chunk_k,
            seqlen_k,
        )

    @cute.jit
    def _fake_to_smem(
        self,
        mdKVaccum_cur: cute.Tensor,
        sdKV: cute.Tensor,
        tidx: cutlass.Int32,
        n_block: cutlass.Int32,
        chunk_k: cutlass.Int32,
        seqlen_k: cutlass.Int32,
    ):
        """Phase 1: gmem f32 (fake-col) -> cvt f32->bf16 -> smem (real-col).

        128 threads each handle one fake_col (= tidx) across all tile_n rows.
        Adjacent threads read adjacent fake-col positions -> coalesced gmem reads
        (128 threads x 4 bytes = 512 bytes = 4 cache lines per row).
        """
        fake_col = tidx

        # fake_col = nt*16 + rank*4 + k
        # real_col = nt*16 + rank*2 + (k//2)*8 + (k%2)
        nt = fake_col // 16
        rank = (fake_col % 16) // 4
        k = fake_col % 4
        real_col = nt * 16 + rank * 2 + (k // 2) * 8 + (k % 2)

        for row in cutlass.range(self.tile_n, unroll_full=True):
            kv_row = n_block * self.tile_n + row
            if kv_row < seqlen_k:
                fake_addr = kv_row * self.head_dim_rounded + chunk_k * self.hdim_chunk + fake_col
                val_f32 = mdKVaccum_cur[fake_addr]
                sdKV[row, real_col] = self.dtype(val_f32)
            else:
                sdKV[row, real_col] = self.dtype(0)

    @cute.jit
    def _smem_to_gmem(
        self,
        sdKV: cute.Tensor,
        mdKV_cur: cute.Tensor,
        tidx: cutlass.Int32,
        n_block: cutlass.Int32,
        chunk_k: cutlass.Int32,
        seqlen_k: cutlass.Int32,
    ):
        """Phase 2: smem (real-col, row-major) -> coalesced gmem bf16 write.

        Strided indexing (tidx + i * num_threads) ensures consecutive threads
        write to consecutive columns within the same row -> coalesced gmem writes
        (128 threads x 2 bytes = 256 bytes = 2 cache lines per row).
        """
        ELEMS_PER_THR = const_expr(self.tile_n * self.hdim_chunk // self.num_threads)

        for i in cutlass.range(ELEMS_PER_THR, unroll_full=True):
            flat_idx = tidx + i * self.num_threads
            local_row = flat_idx // self.hdim_chunk
            col = flat_idx % self.hdim_chunk
            kv_row = n_block * self.tile_n + local_row
            if kv_row < seqlen_k:
                val = sdKV[local_row, col]
                mdKV_cur[kv_row, chunk_k * self.hdim_chunk + col] = val


__all__ = [
    "FlashAttentionDSABackwardSm90",
]
