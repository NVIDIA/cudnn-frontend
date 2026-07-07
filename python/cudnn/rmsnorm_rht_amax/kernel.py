# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""CUTE DSL kernel for fused RMSNorm + RHT + per-CTA amax."""

from __future__ import annotations

import math
import operator
from typing import Any, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.arch import shuffle_sync_bfly
from cutlass.cutlass_dsl import T, dsl_user_op

from .. import data_type
from .._op_kernel import OpKernel
from .._tensor_desc import TensorDesc, make_compact_tensor_desc

DEFAULT_NUM_THREADS_BY_N = {
    2048: 128,
    4096: 256,
    7168: 128,
    8192: 512,
    16384: 1024,
    32768: 512,
}
RPC_CANDIDATES = (2, 4, 8)
TARGET_MIN_CTAS = 148


def best_num_threads(n: int) -> Optional[int]:
    for num_threads in (1024, 512, 256, 128, 64):
        if n % num_threads != 0:
            continue
        ept = n // num_threads
        if ept >= 8 and ept % 8 == 0:
            return num_threads
    return None


def pick_rows_per_cta(m: int) -> int:
    for rows_per_cta in reversed(RPC_CANDIDATES):
        if m % rows_per_cta != 0:
            continue
        if m // rows_per_cta >= TARGET_MIN_CTAS:
            return rows_per_cta
    return RPC_CANDIDATES[0]


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


class RMSNormRHTAmaxKernel(OpKernel):
    """Fused RMSNorm + block-diagonal Hadamard + running per-CTA amax.

    Framework adapters bind generic tensor descriptors, call
    :meth:`check_support`, and materialize the descriptors returned by
    :meth:`infer_output` in their own framework.
    """

    COPY_BITS = 128
    HAD_BLOCK = 16
    TENSOR_ALIGNMENT = COPY_BITS // 8
    output_dtype = data_type.BFLOAT16
    amax_dtype = data_type.FLOAT

    def __init__(
        self,
        *,
        x: TensorDesc[Any],
        weight: TensorDesc[Any],
        output: Optional[TensorDesc[Any]] = None,
        amax: Optional[TensorDesc[Any]] = None,
        eps: float = 1e-5,
        num_threads: Optional[int] = None,
        rows_per_cta: Optional[int] = None,
    ) -> None:
        for name, desc in (("x", x), ("weight", weight)):
            if not isinstance(desc, TensorDesc):
                raise TypeError(f"{name} must be a TensorDesc, got {type(desc).__name__}")
        for name, desc in (("output", output), ("amax", amax)):
            if desc is not None and not isinstance(desc, TensorDesc):
                raise TypeError(f"{name} must be a TensorDesc, got {type(desc).__name__}")

        self.x = x
        self.weight = weight
        self.output = output
        self.amax = amax
        self.eps = eps
        self.requested_num_threads = num_threads
        self.requested_rows_per_cta = rows_per_cta

        self.m: Optional[int] = None
        self.n: Optional[int] = None
        self.num_threads: Optional[int] = None
        self.rows_per_cta: Optional[int] = None

    def check_support(self) -> bool:
        """Validate descriptors and resolve the kernel launch configuration."""

        self.m = None
        self.n = None
        self.num_threads = None
        self.rows_per_cta = None

        if self.x.ndim != 2:
            raise ValueError(f"X must have rank 2, got shape {self.x.shape}")
        if self.weight.ndim != 1:
            raise ValueError(f"W must have rank 1, got shape {self.weight.shape}")
        if self.x.cudnn_dtype != data_type.BFLOAT16:
            raise ValueError(f"X must have dtype bfloat16, got {self.x.dtype}")
        if self.weight.cudnn_dtype != data_type.BFLOAT16:
            raise ValueError(f"W must have dtype bfloat16, got {self.weight.dtype}")

        m, n = self.x.shape
        if m <= 0:
            raise ValueError(f"M must be positive, got {m}")
        if n <= 0:
            raise ValueError(f"N must be positive, got {n}")
        if self.weight.shape != (n,):
            raise ValueError(f"W must have shape {(n,)}, got {self.weight.shape}")
        if self.x.stride != (n, 1) or self.x.stride_order != (1, 0):
            raise ValueError(f"X must be row-major contiguous, got stride {self.x.stride} " f"and stride order {self.x.stride_order}")
        if self.weight.stride != (1,) or self.weight.stride_order != (0,):
            raise ValueError(f"W must be contiguous, got stride {self.weight.stride} " f"and stride order {self.weight.stride_order}")
        if n % self.HAD_BLOCK != 0:
            raise ValueError(f"N must be divisible by {self.HAD_BLOCK} for the Hadamard block size, got {n}")

        num_threads = self.requested_num_threads
        if num_threads is None:
            num_threads = DEFAULT_NUM_THREADS_BY_N.get(n, best_num_threads(n))
        if num_threads is None:
            raise ValueError(f"No valid num_threads found for N={n}")

        rows_per_cta = self.requested_rows_per_cta
        if rows_per_cta is None:
            rows_per_cta = pick_rows_per_cta(m)

        self._validate_launch_configuration(n, num_threads, rows_per_cta, m=m)

        expected_output_shape = (m, n)
        expected_amax_shape = (m // rows_per_cta,)
        if self.output is not None:
            if self.output.shape != expected_output_shape:
                raise ValueError(f"O must have shape {expected_output_shape}, got {self.output.shape}")
            if self.output.cudnn_dtype != self.output_dtype:
                raise ValueError(f"O must have dtype bfloat16, got {self.output.dtype}")
            if self.output.stride != (n, 1) or self.output.stride_order != (1, 0):
                raise ValueError(f"O must be row-major contiguous, got stride {self.output.stride} " f"and stride order {self.output.stride_order}")
        if self.amax is not None:
            if self.amax.shape != expected_amax_shape:
                raise ValueError(f"Amax must have shape {expected_amax_shape}, got {self.amax.shape}")
            if self.amax.cudnn_dtype != self.amax_dtype:
                raise ValueError(f"Amax must have dtype float32, got {self.amax.dtype}")

        self.m = m
        self.n = n
        self.num_threads = num_threads
        self.rows_per_cta = rows_per_cta
        self._configure_lowering_state()
        return True

    def infer_output(self) -> tuple[TensorDesc[data_type], ...]:
        """Infer compact output descriptors from validated input metadata."""

        if self.m is None or self.n is None or self.rows_per_cta is None:
            raise RuntimeError("check_support() must be called before inferring outputs")

        return (
            make_compact_tensor_desc(
                dtype=self.output_dtype,
                shape=(self.m, self.n),
                name="output",
            ),
            make_compact_tensor_desc(
                dtype=self.amax_dtype,
                shape=(self.m // self.rows_per_cta,),
                name="amax",
            ),
        )

    @staticmethod
    def _validate_launch_configuration(
        n: int,
        num_threads: int,
        rows_per_cta: int,
        *,
        m: Optional[int] = None,
    ) -> None:
        if n <= 0:
            raise ValueError(f"N must be positive, got {n}")
        if num_threads <= 0:
            raise ValueError(f"num_threads must be positive, got {num_threads}")
        if num_threads % 32 != 0:
            raise ValueError(f"num_threads must be warp-aligned, got {num_threads}")
        if num_threads > 1024:
            raise ValueError(f"num_threads must not exceed the CUDA block size limit, got {num_threads}")
        if n % num_threads != 0:
            raise ValueError(f"N={n} must be divisible by num_threads={num_threads}")

        ept = n // num_threads
        if ept < 8 or ept % 8 != 0:
            raise ValueError(f"EPT={ept} must be >= 8 and divisible by 8")
        if rows_per_cta <= 0:
            raise ValueError(f"rows_per_cta must be positive, got {rows_per_cta}")
        if m is not None and m % rows_per_cta != 0:
            raise ValueError(f"M must be divisible by rows_per_cta, got M={m}, rows_per_cta={rows_per_cta}")

    def _configure_lowering_state(self) -> None:
        if self.n is None or self.num_threads is None or self.rows_per_cta is None:
            raise RuntimeError("Kernel launch configuration has not been resolved")

        self._validate_launch_configuration(self.n, self.num_threads, self.rows_per_cta, m=self.m)
        self.vec_size = self.COPY_BITS // 16
        self.ept = self.n // self.num_threads

        self.num_vec_blocks = self.ept // self.vec_size
        self.warps_per_row = self.num_threads // 32
        self.inv_sqrt_had = 1.0 / math.sqrt(self.HAD_BLOCK)
        self.num_intra_stages = int(math.log2(self.vec_size))
        self.num_cross_stages = 1

        self.tv_shape = ((self.num_threads, 1), (self.vec_size, self.num_vec_blocks))
        self.tv_stride = ((self.vec_size, 1), (1, self.vec_size * self.num_threads))
        self.tiler_mn = (1, self.n)

        tile_bytes = self.n * 2
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
    def __call__(
        self,
        x_tensor: cute.Tensor,
        w_tensor: cute.Tensor,
        o_tensor: cute.Tensor,
        amax_tensor: cute.Tensor,
        stream: cuda.CUstream,
    ):
        m = x_tensor.shape[0]
        num_ctas = m // self.rows_per_cta
        tv_layout = cute.make_layout(self.tv_shape, stride=self.tv_stride)
        self.kernel(
            x_tensor,
            w_tensor,
            o_tensor,
            amax_tensor,
            Float32(self.eps),
            tv_layout,
            self.tiler_mn,
        ).launch(
            grid=(num_ctas, 1, 1),
            block=(self.num_threads, 1, 1),
            smem=self.smem_bytes,
            stream=stream,
        )
