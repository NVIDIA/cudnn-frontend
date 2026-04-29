# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""CUTE DSL kernel for fused RMSNorm + RHT + per-CTA amax."""

import math
import operator

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.arch import shuffle_sync_bfly
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def fabs_f32(val, *, loc=None, ip=None):
    val_ir = val.ir_value(loc=loc, ip=ip)
    result = llvm.inline_asm(
        T.f32(),
        [val_ir],
        "abs.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Float32(result)


@dsl_user_op
def fmax_f32(a, b, *, loc=None, ip=None):
    a_ir = a.ir_value(loc=loc, ip=ip)
    b_ir = b.ir_value(loc=loc, ip=ip)
    result = llvm.inline_asm(
        T.f32(),
        [a_ir, b_ir],
        "max.f32 $0, $1, $2;",
        "=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Float32(result)


@dsl_user_op
def redux_sync_max_f32(val, *, loc=None, ip=None):
    val_ir = val.ir_value(loc=loc, ip=ip)
    result = llvm.inline_asm(
        T.f32(),
        [val_ir],
        "redux.sync.max.f32 $0, $1, 0xffffffff;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Float32(result)


class RMSNormRHTAmaxKernel:
    """Fused RMSNorm + block-diagonal Hadamard + running per-CTA amax."""

    COPY_BITS = 128
    HAD_BLOCK = 16

    def __init__(self, n, num_threads=256, eps=1e-5, rows_per_cta=8):
        self.n = n
        self.num_threads = num_threads
        self.eps = eps
        self.rows_per_cta = rows_per_cta
        self.vec_size = self.COPY_BITS // 16
        self.ept = n // num_threads

        assert n % num_threads == 0, f"N={n} must be divisible by num_threads={num_threads}"
        assert self.ept % self.vec_size == 0, f"EPT={self.ept} must be a multiple of vec_size={self.vec_size}"
        assert self.ept >= self.vec_size, f"EPT={self.ept} must be >= vec_size={self.vec_size}"

        self.num_vec_blocks = self.ept // self.vec_size
        self.warps_per_row = num_threads // 32
        self.inv_sqrt_had = 1.0 / math.sqrt(self.HAD_BLOCK)
        self.num_intra_stages = int(math.log2(self.vec_size))
        self.num_cross_stages = 1

        self.tv_shape = ((num_threads, 1), (self.vec_size, self.num_vec_blocks))
        self.tv_stride = ((self.vec_size, 1), (1, self.vec_size * num_threads))
        self.tiler_mn = (1, n)

        tile_bytes = n * 2
        reduce_bytes = self.warps_per_row * 4
        amax_bytes = self.warps_per_row * 4
        self.smem_bytes = tile_bytes + reduce_bytes + amax_bytes + 128

        self.intra_butterfly_pairs = []
        for stage in range(self.num_intra_stages):
            delta = 1 << stage
            pairs = []
            for pair_idx in range(self.vec_size // 2):
                i_idx = (pair_idx // delta) * 2 * delta + (pair_idx % delta)
                j_idx = i_idx + delta
                pairs.append((i_idx, j_idx))
            self.intra_butterfly_pairs.append(pairs)

    @cute.kernel
    def kernel(self, m_x: cute.Tensor, m_w: cute.Tensor, m_o: cute.Tensor, m_amax: cute.Tensor, eps: Float32, tv_layout: cute.Layout, tiler_mn: cute.Shape):
        cfg = self
        tid = cute.arch.thread_idx()[0]
        bid = cute.arch.block_idx()[0]
        inv_sqrt_had = cutlass.Float32(cfg.inv_sqrt_had)

        smem = utils.SmemAllocator()
        s_x = smem.allocate_tensor(
            cutlass.BFloat16,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer = smem.allocate_tensor(Float32, cute.make_layout((1, cfg.warps_per_row)), byte_alignment=4)
        amax_buffer = smem.allocate_tensor(Float32, cute.make_layout((1, cfg.warps_per_row)), byte_alignment=4)

        copy_atom_g2s = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=cfg.COPY_BITS)
        copy_atom_load_w = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=cfg.COPY_BITS)
        copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=cfg.COPY_BITS)

        tiled_copy_load = cute.make_tiled_copy(copy_atom_g2s, tv_layout, tiler_mn)
        tiled_copy_w = cute.make_tiled_copy(copy_atom_load_w, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_load = tiled_copy_load.get_slice(tid)
        thr_w = tiled_copy_w.get_slice(tid)
        thr_store = tiled_copy_store.get_slice(tid)

        t_xs_x = thr_load.partition_D(s_x)

        m_w_layout = cute.prepend(m_w.layout, cute.make_layout((1,), stride=(0,)))
        m_w_2d = cute.make_tensor(m_w.iterator, m_w_layout)
        g_w = cute.local_tile(m_w_2d, tiler_mn, (0, 0))
        t_wg_w = thr_w.partition_S(g_w)
        t_wr_w = cute.make_fragment_like(t_wg_w)
        cute.copy(copy_atom_load_w, t_wg_w, t_wr_w)
        t_xr_w = thr_load.retile(t_wr_w)

        row_base = bid * cfg.rows_per_cta
        g_x_first = cute.local_tile(m_x, tiler_mn, (row_base, 0))
        t_xg_x_first = thr_load.partition_S(g_x_first)
        t_xr_x = cute.make_fragment_like(t_xg_x_first)

        cute.copy(copy_atom_g2s, t_xg_x_first, t_xs_x)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        reg = cute.make_rmem_tensor(cute.make_layout((cfg.ept,)), cutlass.Float32)
        lane_id = cute.arch.lane_idx()
        warp_id = cute.arch.warp_idx()
        running_max = cutlass.Float32(0.0)

        for row_idx in cutlass.range_constexpr(cfg.rows_per_cta):
            cute.autovec_copy(t_xs_x, t_xr_x)

            if row_idx < cfg.rows_per_cta - 1:
                g_x_next = cute.local_tile(m_x, tiler_mn, (row_base + (row_idx + 1), 0))
                t_xg_x_next = thr_load.partition_S(g_x_next)
                cute.copy(copy_atom_g2s, t_xg_x_next, t_xs_x)
                cute.arch.cp_async_commit_group()

            x = t_xr_x.load().to(Float32)
            x_sq = x * x
            local_sum = x_sq.reduce(cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0)
            warp_sum = cute.arch.warp_reduction(local_sum, operator.add)
            if lane_id == 0:
                reduction_buffer[0, warp_id] = warp_sum
            cute.arch.barrier()

            block_val = Float32(0.0)
            if lane_id < cfg.warps_per_row:
                block_val = reduction_buffer[0, lane_id]
            sum_sq = cute.arch.warp_reduction(block_val, operator.add)

            mean_sq = sum_sq / cfg.n
            rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

            w = t_xr_w.load().to(Float32)
            y = x * rstd * w

            for elem_idx in cutlass.range_constexpr(cfg.ept):
                reg[elem_idx] = y[elem_idx]

            for block_idx in cutlass.range_constexpr(cfg.num_vec_blocks):
                block_offset = block_idx * cfg.vec_size
                for stage_idx in cutlass.range_constexpr(cfg.num_intra_stages):
                    for pair_idx in cutlass.range_constexpr(cfg.vec_size // 2):
                        i_idx = block_offset + cfg.intra_butterfly_pairs[stage_idx][pair_idx][0]
                        j_idx = block_offset + cfg.intra_butterfly_pairs[stage_idx][pair_idx][1]
                        a_val = reg[i_idx]
                        b_val = reg[j_idx]
                        reg[i_idx] = a_val + b_val
                        reg[j_idx] = a_val - b_val

            for cross_stage in cutlass.range_constexpr(cfg.num_cross_stages):
                xor_mask = cutlass.Int32(1 << cross_stage)
                is_lower = (tid & xor_mask) == cutlass.Int32(0)
                for elem_idx in cutlass.range_constexpr(cfg.ept):
                    partner = shuffle_sync_bfly(reg[elem_idx], offset=xor_mask)
                    if is_lower:
                        reg[elem_idx] = reg[elem_idx] + partner
                    else:
                        reg[elem_idx] = partner - reg[elem_idx]

            for elem_idx in cutlass.range_constexpr(cfg.ept):
                scaled = reg[elem_idx] * inv_sqrt_had
                abs_val = fabs_f32(scaled)
                running_max = fmax_f32(running_max, abs_val)
                t_xr_x[elem_idx] = scaled.to(cutlass.BFloat16)

            g_o_r = cute.local_tile(m_o, tiler_mn, (row_base + row_idx, 0))
            t_xg_o_r = thr_store.partition_D(g_o_r)
            cute.copy(copy_atom_store, t_xr_x, t_xg_o_r)

            if row_idx < cfg.rows_per_cta - 1:
                cute.arch.cp_async_wait_group(0)

        warp_max = redux_sync_max_f32(running_max)
        if lane_id == 0:
            amax_buffer[0, warp_id] = warp_max
        cute.arch.barrier()

        amax_val = cutlass.Float32(0.0)
        if lane_id < cfg.warps_per_row:
            amax_val = amax_buffer[0, lane_id]
        cta_max = redux_sync_max_f32(amax_val)
        if tid == cutlass.Int32(0):
            m_amax[bid] = cta_max

    @cute.jit
    def __call__(self, x_tensor: cute.Tensor, w_tensor: cute.Tensor, o_tensor: cute.Tensor, amax_tensor: cute.Tensor, eps: Float32, stream: cuda.CUstream):
        m = x_tensor.shape[0]
        num_ctas = m // self.rows_per_cta
        tv_layout = cute.make_layout(self.tv_shape, stride=self.tv_stride)
        self.kernel(x_tensor, w_tensor, o_tensor, amax_tensor, eps, tv_layout, self.tiler_mn).launch(
            grid=(num_ctas, 1, 1),
            block=(self.num_threads, 1, 1),
            smem=self.smem_bytes,
            stream=stream,
        )
