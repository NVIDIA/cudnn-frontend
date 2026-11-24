# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""SM100-family BF16 GEMM. Each MMA group owns an output tile and one contiguous K chunk.

Adapted from FlashInfer's batch-invariant BF16 GEMM. Each chunk folds ascending K16
MMA instructions into an FP32 accumulator. Chunk boundaries depend only on N and K.
FP32 chunk sums and the optional BF16 bias are added in order before the final cast.
"""

from dataclasses import dataclass
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.math import RoundingMode
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.tensor_utils import LayoutEnum
from cutlass.utils.blackwell_helpers import get_smem_store_op, get_tmem_load_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from cudnn.frost.compiled_cache import compile_cached
from cudnn.frost.device import device_context
from cudnn.frost.tile_dsl.barrier import named_barrier_sync_unaligned

from . import chunks_for, schedule_for

_THREADS = 192
_MMA_WARP = 4
_TMA_WARP = 5


class _Gemm:
    def __init__(self, b_major, chunks, biased, tile, ab_stages, group=1, cluster_n=1):
        self.cluster = (group, cluster_n)
        self.group = group
        self.block_tile = (tile[0] // group, tile[1], tile[2])
        self.tile = tile
        self.ab_stages = ab_stages
        self.b_major = b_major
        self.chunks = chunks
        self.biased = biased

    @cute.jit
    def __call__(
        self,
        a_addr: Int64,
        b_addr: Int64,
        out_addr: Int64,
        bias_addr: Int64,
        workspace_addr: Int64,
        m: Int32,
        n: Int32,
        k: Int32,
        lda: Int64,
        stream: cuda.CUstream,
    ):
        chunks = self.chunks
        kc = cute.assume(k // chunks, divby=8)
        n = cute.assume(n, divby=8)
        row_pitch = cute.assume(lda, divby=8)
        k_stride = cute.assume(Int64(k), divby=8)
        n_stride = cute.assume(Int64(n), divby=8)
        kc_stride = cute.assume(Int64(kc), divby=8)
        a = cute.make_tensor(
            cute.make_ptr(BFloat16, a_addr, cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((m, kc, chunks), stride=(row_pitch, 1, kc_stride)),
        )
        # Graph B has shape (K,N); the MMA instruction consumes the transposed (N,K) view.
        b_strides = (k_stride, 1, kc_stride) if self.b_major == "k" else (1, n_stride, kc_stride * n_stride)
        b = cute.make_tensor(
            cute.make_ptr(BFloat16, b_addr, cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((n, kc, chunks), stride=b_strides),
        )
        out = cute.make_tensor(
            cute.make_ptr(BFloat16, out_addr, cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((m, n, 1), stride=(n_stride, 1, 0)),
        )
        bias = None
        if cutlass.const_expr(self.biased):
            bias = cute.make_tensor(cute.make_ptr(BFloat16, bias_addr, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout(n))
        partials = None
        if cutlass.const_expr(chunks > 1):
            partials = cute.make_tensor(
                cute.make_ptr(Float32, workspace_addr, cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout((m, n, chunks), stride=(n_stride, 1, Int64(m) * n_stride)),
            )
            self._launch(a, b, partials, None, stream)
            rows = cutlass.min(m, Int32(65535))
            self._reduce(partials, out, bias, rows).launch(grid=(cute.ceil_div(n, 256), rows, 1), block=(256, 1, 1), stream=stream)
        else:
            self._launch(a, b, out, bias, stream)

    def _launch(self, a, b, c, bias, stream):
        mma = utils.sm100.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            LayoutEnum.from_tensor(a).mma_major_mode(),
            LayoutEnum.from_tensor(b).mma_major_mode(),
            Float32,
            tcgen05.CtaGroup.TWO if self.group == 2 else tcgen05.CtaGroup.ONE,
            self.tile[:2],
        )
        cluster = cute.tiled_divide(cute.make_layout((*self.cluster, 1)), (mma.thr_id.shape,))
        smem_a = utils.sm100.make_smem_layout_a(mma, self.tile, BFloat16, self.ab_stages)
        smem_b = utils.sm100.make_smem_layout_b(mma, self.tile, BFloat16, self.ab_stages)
        stage_a = cute.select(smem_a, mode=[0, 1, 2])
        stage_b = cute.select(smem_b, mode=[0, 1, 2])
        op_a = utils.sm100.cluster_shape_to_tma_atom_A(self.cluster, mma.thr_id)
        op_b = utils.sm100.cluster_shape_to_tma_atom_B(self.cluster, mma.thr_id)
        atom_a, gA = cute.nvgpu.make_tiled_tma_atom_A(op_a, a, stage_a, self.tile, mma, cluster.shape)
        atom_b, gB = cute.nvgpu.make_tiled_tma_atom_B(op_b, b, stage_b, self.tile, mma, cluster.shape)
        c_layout = LayoutEnum.from_tensor(c)
        epi_tile = utils.sm100.compute_epilogue_tile_shape(self.block_tile, self.group == 2, c_layout, c.element_type)
        self.c_layout = c_layout
        self.c_dtype = c.element_type
        smem_c = utils.sm100.make_smem_layout_epi(c.element_type, c_layout, epi_tile, 2)
        atom_c, gC = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), c, cute.select(smem_c, mode=[0, 1]), epi_tile)
        self.tx_bytes = self.group * (cute.size_in_bytes(BFloat16, stage_a) + cute.size_in_bytes(BFloat16, stage_b))
        tm = (c.shape[0] - 1) // self.tile[0] + 1
        n_clusters = (c.shape[1] - 1) // (self.tile[1] * self.cluster[1]) + 1
        self._tile(mma, cluster, atom_a, gA, atom_b, gB, atom_c, gC, bias, smem_a, smem_b, smem_c, epi_tile, tm).launch(
            grid=(tm * n_clusters * self.group, self.cluster[1], self.chunks), block=(_THREADS, 1, 1), cluster=(*self.cluster, 1), stream=stream
        )

    @cute.kernel
    def _tile(
        self,
        mma,
        cluster: cute.Layout,
        atom_a: cute.CopyAtom,
        gA: cute.Tensor,
        atom_b: cute.CopyAtom,
        gB: cute.Tensor,
        atom_c: cute.CopyAtom,
        c: cute.Tensor,
        bias: Optional[cute.Tensor],
        smem_a: cute.ComposedLayout,
        smem_b: cute.ComposedLayout,
        smem_c: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tm: Int32,
    ):
        @cute.struct
        class SharedStorage:
            ab_barriers: cute.struct.MemRange[Int64, 2 * self.ab_stages]
            acc_barriers: cute.struct.MemRange[Int64, 2]
            dealloc_barrier: Int64
            tmem_addr: Int32

        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tid, _, _ = cute.arch.thread_idx()
        bx, by, chunk = cute.arch.block_idx()
        tile_idx = bx // self.group
        m_tile, n_tile = tile_idx % tm, tile_idx // tm * self.cluster[1] + by
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        coord = cluster.get_flat_coord(rank)
        rank_mma = bx % self.group
        leader = rank_mma == 0
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        producer, consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_barriers.data_ptr(),
            num_stages=self.ab_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Warp),
            tx_count=self.tx_bytes,
            cta_layout_vmnk=cluster,
            enable_multicast_signaling=True,
            defer_sync=True,
        ).make_participants()
        acc_pipe = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_barriers.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4 * self.group),
            cta_layout_vmnk=cluster,
            defer_sync=True,
        )
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_addr.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=2, num_threads=_THREADS),
            allocator_warp_id=0,
            is_two_cta=self.group == 2,
            two_cta_tmem_dealloc_mbar_ptr=storage.dealloc_barrier.ptr,
        )
        pipeline.pipeline_init_arrive(cluster_shape_mn=self.cluster, is_relaxed=True)
        sA = smem.allocate_tensor(BFloat16, layout=smem_a.outer, byte_alignment=128, swizzle=smem_a.inner)
        sB = smem.allocate_tensor(BFloat16, layout=smem_b.outer, byte_alignment=128, swizzle=smem_b.inner)
        sC = smem.allocate_tensor(self.c_dtype, layout=smem_c.outer, byte_alignment=128, swizzle=smem_c.inner)
        gA_tile = cute.local_tile(gA, (self.tile[0], self.tile[2]), (m_tile, None, chunk))
        gB_tile = cute.local_tile(gB, (self.tile[1], self.tile[2]), (n_tile, None, chunk))
        thr_mma = mma.get_slice(rank_mma)
        a_cluster = cute.make_layout(cute.slice_(cluster, (0, 0, None, 0)).shape)
        b_cluster = cute.make_layout(cute.slice_(cluster, (0, None, 0, 0)).shape)
        mask_a, mask_b = None, None
        if cutlass.const_expr(self.cluster != (1, 1)):
            mask_a = cpasync.create_tma_multicast_mask(cluster, coord, mcast_mode=2)
            mask_b = cpasync.create_tma_multicast_mask(cluster, coord, mcast_mode=1)
        tAsA, tAgA = cpasync.tma_partition(atom_a, coord[2], a_cluster, cute.group_modes(sA, 0, 3), cute.group_modes(thr_mma.partition_A(gA_tile), 0, 3))
        tBsB, tBgB = cpasync.tma_partition(atom_b, coord[1], b_cluster, cute.group_modes(sB, 0, 3), cute.group_modes(thr_mma.partition_B(gB_tile), 0, 3))
        rA = mma.make_fragment_A(sA)
        rB = mma.make_fragment_B(sB)
        fake_acc = mma.make_fragment_C(thr_mma.partition_shape_C(self.tile[:2]))
        num_cols = cutlass.memory.get_num_tmem_alloc_cols(fake_acc, arch="sm_100")
        num_k_tiles = cute.size(gA_tile, mode=[2])
        pipeline.pipeline_init_wait(cluster_shape_mn=self.cluster)
        # All warps meet at one allocation barrier before they take their roles.
        tmem.allocate(num_cols)
        tmem.wait_for_alloc()
        ptr = tmem.retrieve_ptr(Float32)
        acc = cute.make_tensor(ptr, fake_acc.layout)

        if warp == _TMA_WARP:
            empty = producer.try_acquire()
            for k_tile in cutlass.range(num_k_tiles, unroll=1):
                handle = producer.acquire_and_advance(empty)
                cute.copy(atom_a, tAgA[None, k_tile], tAsA[None, handle.index], tma_bar_ptr=handle.barrier, mcast_mask=mask_a)
                cute.copy(atom_b, tBgB[None, k_tile], tBsB[None, handle.index], tma_bar_ptr=handle.barrier, mcast_mask=mask_b)
                empty = cutlass.Boolean(1)
                if handle.count + 1 < num_k_tiles:
                    empty = producer.try_acquire()
            producer.tail()
        if warp == _MMA_WARP and leader:
            state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            acc_pipe.producer_acquire(state)
            full = consumer.try_wait()
            for k_tile in cutlass.range(num_k_tiles, unroll=1):
                handle = consumer.wait_and_advance(full)
                for j in cutlass.range_constexpr(self.tile[2] // 16):
                    # The first K16 instruction starts a fresh accumulator for this chunk.
                    mma.set(tcgen05.Field.ACCUMULATE, (k_tile != 0) | (j != 0))
                    crd = (None, None, j, handle.index)
                    cute.gemm(mma, acc, rA[crd], rB[crd], acc)
                handle.release()
                full = cutlass.Boolean(1)
                if handle.count + 1 < num_k_tiles:
                    full = consumer.try_wait()
            acc_pipe.producer_commit(state)
        if warp < _MMA_WARP:
            state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            acc_pipe.consumer_wait(state)
            self._store(acc, atom_c, c, sC, epi_tile, bias, thr_mma, tid, m_tile, n_tile, chunk)
            cute.arch.fence_view_async_tmem_load()
        # A block owns exactly one tile. Its TMEM columns live through the last load by every epilogue warp.
        cute.arch.sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(ptr)

    @cute.jit
    def _store(self, acc, atom_c, c, sC, epi_tile, bias, thr_mma, tid, m_tile, n_tile, chunk):
        copy_op = get_tmem_load_op(self.block_tile, self.c_layout, self.c_dtype, Float32, epi_tile, self.group == 2)
        src = cute.flat_divide(transform_partitioned_tensor_layout(acc), epi_tile)
        copy = tcgen05.make_tmem_copy(copy_op, src[None, None, 0, 0])
        gC = cute.local_tile(c, self.tile[:2], (m_tile, n_tile, chunk))
        dest = cute.flat_divide(transform_partitioned_tensor_layout(thr_mma.partition_C(gC)), epi_tile)
        tSsC, tSgC = cpasync.tma_partition(atom_c, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), cute.group_modes(dest, 0, 2))
        tSgC = cute.group_modes(tSgC, 1, cute.rank(tSgC))
        src = cute.group_modes(copy.get_slice(tid).partition_S(src), 3, 5)
        reg_shape = copy.get_slice(tid).partition_D(dest)[None, None, None, 0, 0].shape
        frag = cute.make_rmem_tensor(reg_shape, Float32)
        result = cute.make_rmem_tensor(reg_shape, self.c_dtype)
        to_smem = cute.make_tiled_copy_D(get_smem_store_op(self.c_layout, self.c_dtype, Float32, copy), copy)
        reg = to_smem.retile(result)
        smem = to_smem.get_slice(tid).partition_D(sC)
        stores = pipeline.PipelineTmaStore.create(num_stages=2, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128))
        coords = None
        if cutlass.const_expr(bias is not None):
            identity = cute.local_tile(cute.make_identity_tensor(c.shape), self.tile[:2], (m_tile, n_tile, chunk))
            coords = cute.flat_divide(transform_partitioned_tensor_layout(thr_mma.partition_C(identity)), epi_tile)
            coords = cute.group_modes(copy.get_slice(tid).partition_D(coords), 3, 5)
        for s in cutlass.range_constexpr(cute.size(src, mode=[3])):
            cute.copy(copy, src[None, None, None, s], frag)
            if cutlass.const_expr(bias is not None):
                for i in cutlass.range_constexpr(cute.size(frag)):
                    col = coords[None, None, None, s][i][1]
                    if col < c.shape[1]:
                        frag[i] = cute.math.add(frag[i], bias[col].to(Float32), rounding=RoundingMode.NEAREST_EVEN)
            reg.store(to_smem.retile(frag).load().to(self.c_dtype))
            cute.copy(to_smem, reg, smem[None, None, None, s % 2])
            cute.arch.fence_proxy("async.shared", space="cta")
            named_barrier_sync_unaligned(3, 128)
            if tid < 32:
                cute.copy(atom_c, tSsC[None, s % 2], tSgC[None, s])
                stores.producer_commit()
                stores.producer_acquire()
            named_barrier_sync_unaligned(3, 128)
        if tid < 32:
            stores.producer_tail()

    @cute.kernel
    def _reduce(self, partials: cute.Tensor, out: cute.Tensor, bias: Optional[cute.Tensor], rows: Int32):
        block_x, block_y, _ = cute.arch.block_idx()
        col = block_x * 256 + cute.arch.thread_idx()[0]
        if col < out.shape[1]:
            for row in cutlass.range(block_y, out.shape[0], rows, unroll=1):
                value = Float32(0)
                for chunk in cutlass.range_constexpr(self.chunks):
                    value = cute.math.add(value, partials[row, col, chunk], rounding=RoundingMode.NEAREST_EVEN)
                if cutlass.const_expr(bias is not None):
                    value = cute.math.add(value, bias[col].to(Float32), rounding=RoundingMode.NEAREST_EVEN)
                out[row, col, 0] = value.to(BFloat16)


@dataclass(frozen=True)
class _Compiled:
    launch: object
    m: int
    n: int
    k: int
    lda: int
    workspace_bytes: int

    def execute(self, a, b, out, bias, workspace, stream):
        self.launch(a, b, out, bias or 0, workspace or 0, self.m, self.n, self.k, self.lda, cuda.CUstream(stream))


def build(m, n, k, lda, b_major, bias, device):
    chunks = chunks_for(n, k)
    cfg = schedule_for(m, n, chunks)
    tile = (cfg.tile_m, cfg.tile_n, 64)
    with device_context(device):
        launch = compile_cached(
            _Gemm(b_major, chunks, bias, tile, cfg.stages, cfg.mma_group, cfg.cluster_n),
            *([Int64(0)] * 5),
            Int32(0),
            Int32(0),
            Int32(0),
            Int64(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            cache_key=f"rl_gemm_bf16_{b_major}_{chunks}_{int(bias)}_{tile[0]}x{tile[1]}_{cfg.stages}_{cfg.mma_group}x{cfg.cluster_n}",
            symbol="rl_gemm_bf16",
            options="--enable-tvm-ffi",
        )
    return _Compiled(launch, m, n, k, lda, chunks * m * n * 4 if chunks > 1 else 0)
