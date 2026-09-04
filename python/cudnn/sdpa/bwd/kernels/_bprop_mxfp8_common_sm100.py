# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Shared CuTe DSL primitives for the SM100 D256 MXFP8 SDPA-backward kernels.

One module = Xinbo's ``fmha_cute/{constants,primitives,d256_primitives}.py``
plus the JIT-side tensor helpers of ``fmha_cute/layouts.py`` (the torch-side
test helpers of that file are NOT ported; the adapter in ``bwd/api_dsl.py``
builds the operand views).

Ported from Xinbo Zhao's fmha_mxfp8_large_head_dim (2026-09-01).
Kept as close to the source as the package rules allow so upstream fixes
stay diff-able; only the imports and this note differ.
"""

from typing import Optional, Type, Tuple, Union

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass.cute.nvgpu import cpasync, tcgen05, OperandMajorMode
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.typing import Int32, Uint32, Float32, Boolean
from cutlass.cute.typing import Int8

# --- constants (fmha_cute/constants.py) ---------------------------------

LOW_PRECISION_TYPE = cutlass.Float8E4M3FN
SF_DTYPE = cutlass.Float8E8M0FNU
SF_VEC_SIZE = 32
E4M3_MAX = 448.0
SM100_TMEM_CAPACITY_COLUMNS = 512


def get_cute_element_dtype(name: str):
    """Map public dtype names to CuTe numeric types."""
    from cutlass.cute.typing import BFloat16, Float16, Float8E4M3FN

    mapping = {
        "Float8E4M3FN": Float8E4M3FN,
        "Float16": Float16,
        "BFloat16": BFloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported CUTE element dtype {name}") from exc


# --- JIT tensor helpers (fmha_cute/layouts.py) ---------------------------


@cute.jit
def make_q_head_batch_tensor(
    tensor: cute.Tensor,
    hb,
    varlen: cutlass.Constexpr,
):
    """Group ``(h_r, h_k, b)`` while preserving every Q head."""
    (h_r, h_k), _ = hb
    batch_stride = (
        0
        if varlen
        else cute.assume(
            tensor.shape[0] * tensor.shape[1] * h_r * h_k,
            divby=64,
        )
    )
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(
            (tensor.shape[0], tensor.shape[1], hb),
            stride=(
                tensor.stride[0],
                tensor.stride[1],
                (
                    (
                        tensor.shape[1],
                        tensor.shape[1] * tensor.shape[2],
                    ),
                    batch_stride,
                ),
            ),
        ),
    )


def make_kv_head_batch_layout(
    tensor: cute.Tensor,
    hb,
    varlen: cutlass.Constexpr,
):
    """Build a grouped KV-head layout inside the caller JIT trace.

    A separate ``@cute.jit`` boundary cannot safely return this dynamic output
    layout for multi-head or multi-batch problems.
    """
    (_, h_k), _ = hb
    batch_stride = (
        0
        if varlen
        else cute.assume(
            tensor.shape[0] * tensor.shape[1] * h_k,
            divby=64,
        )
    )
    return cute.make_layout(
        (tensor.shape[0], tensor.shape[1], hb),
        stride=(
            tensor.stride[0],
            tensor.stride[1],
            ((0, tensor.shape[1]), batch_stride),
        ),
    )


@cute.jit
def make_kv_head_batch_tensor(
    tensor: cute.Tensor,
    hb,
    varlen: cutlass.Constexpr,
):
    """Group ``(1, h_k, b)`` and broadcast the GQA ``h_r`` mode."""
    return cute.make_tensor(
        tensor.iterator,
        make_kv_head_batch_layout(tensor, hb, varlen),
    )


@cute.jit
def make_transposed_tensor(tensor: cute.Tensor, source_layout):
    """View a separately stored operand with the first two modes swapped."""
    return cute.make_tensor(
        tensor.iterator,
        cute.select(source_layout, mode=[1, 0, 2]),
    )


@cute.jit
def make_lse_head_batch_tensor(tensor: cute.Tensor, hb):
    """Group the LSE ``(h_r, h_k, b)`` modes."""
    batch_stride = 0 if tensor.shape[3] == 1 else tensor.shape[0] * tensor.shape[1] * tensor.shape[2]
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(
            (tensor.shape[0], hb),
            stride=(
                tensor.stride[0],
                (
                    (
                        tensor.shape[0],
                        tensor.shape[0] * tensor.shape[1],
                    ),
                    batch_stride,
                ),
            ),
        ),
    )


# --- primitives (fmha_cute/primitives.py) --------------------------------


def init_common_config(self):
    """Initialize config shared between dQ and dKdV kernels.

    Sets up: sum_OdO config, warp IDs, compute warp counts,
    TMEM capacity, threads_per_warp, barriers 1-6, register allocations,
    buffer_align_bytes.

    Call this from __init__ after kernel-specific tiler setup.
    Concrete classes set threads_per_cta and may add extra barriers after this call.
    """
    # =================== Sum OdO ================================
    self.sum_OdO_max_threads_per_block = 128
    self.sum_OdO_block_q = 16
    self.sum_OdO_num_threads_d = 8
    self.sum_OdO_num_threads_q = self.sum_OdO_max_threads_per_block // self.sum_OdO_num_threads_d
    self.sum_OdO_elem_per_load = 2

    # self.reduce_warp_id = (0, 1, 2, 3)
    self.compute_warp_id_0 = (0, 1, 2, 3)
    self.compute_warp_id_1 = (4, 5, 6, 7)
    self.mma_warp_id = 8
    self.load_warp_id = 9

    self.num_compute_0_warps = len(self.compute_warp_id_0)
    self.num_compute_1_warps = len(self.compute_warp_id_1)
    self.num_compute_warps = self.num_compute_0_warps + self.num_compute_1_warps

    SM100_TMEM_CAPACITY_COLUMNS = 512
    self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

    self.threads_per_warp = 32
    # threads_per_cta is set by each concrete kernel.

    self.cta_sync_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=self.threads_per_cta,
    )
    self.tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=2,
        num_threads=self.threads_per_warp,
    )
    self.epilogue_sync_barrier = pipeline.NamedBarrier(
        barrier_id=4,
        num_threads=self.num_compute_0_warps * self.threads_per_warp,
    )
    self.dS_sync_barrier_compute0 = pipeline.NamedBarrier(
        barrier_id=5,
        num_threads=(self.num_compute_0_warps + 1) * self.threads_per_warp,  # +1 for mma warp
    )
    self.dS_sync_barrier_compute1 = pipeline.NamedBarrier(
        barrier_id=6,
        num_threads=(self.num_compute_1_warps + 1) * self.threads_per_warp,  # +1 for mma warp
    )
    self.num_regs_compute = 160
    self.num_regs_mma = 96
    self.num_regs_empty = 96
    self.num_regs_load = 96

    self.buffer_align_bytes = 1024
    self.store_num_bits_per_copy = 128


def reserve_tmem_tensor(
    tmem_ptr,
    tmem_offset,
    layout,
    dtype,
):
    """Bind ``layout`` to the next free TMEM columns.

    The helper is intentionally a plain Python function: CuTe inlines it while
    tracing the caller, so it only removes allocation bookkeeping from the
    kernels and does not add a device-side function call.
    """
    tensor = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + tmem_offset, dtype=dtype),
        layout,
    )
    num_cols = tcgen05.find_tmem_tensor_col_offset(tensor)
    return tensor, tmem_offset + num_cols, num_cols


def reserve_grouped_tmem_tensor(
    tmem_ptr,
    tmem_offset,
    layout,
    dtype,
    group_count,
):
    """Bind repeated ``layout`` instances to one tensor with a group mode."""
    first = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + tmem_offset, dtype=dtype),
        layout,
    )
    group_stride = tcgen05.find_tmem_tensor_col_offset(first)
    # The allocator reports 32-bit TMEM columns, while a layout stride after
    # recasting is measured in ``dtype`` elements.
    assert 32 % dtype.width == 0
    group_element_stride = group_stride * (32 // dtype.width)
    grouped_layout = cute.make_layout(
        (*layout.shape, group_count),
        stride=(*layout.stride, group_element_stride),
    )
    tensor = cute.make_tensor(first.iterator, grouped_layout)
    return tensor, tmem_offset + group_stride * group_count, group_stride


def reserve_tmem_fragment(
    tmem_ptr,
    tmem_offset,
    fragment,
    dtype,
):
    """Bind a pre-built TMEM fragment to the next free columns."""
    num_cols = tcgen05.find_tmem_tensor_col_offset(fragment)
    fragment = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + tmem_offset, dtype=dtype),
        fragment.layout,
    )
    return fragment, tmem_offset + num_cols, num_cols


def reserve_tmem_mma_fragment(
    tmem_ptr,
    tmem_offset,
    tiled_mma,
    mma_tiler,
    num_stages,
    dtype,
):
    """Reserve TMEM columns for an MMA C fragment and bind its base pointer."""
    fragment_shape = tiled_mma.partition_shape_C(cute.append(cute.select(mma_tiler, mode=[0, 1]), num_stages))
    fragment = tiled_mma.make_fragment_C(fragment_shape)
    return reserve_tmem_fragment(tmem_ptr, tmem_offset, fragment, dtype)


# =================== @cute.kernel methods ========================


@cute.kernel
def sum_OdO(
    self,
    O: cute.Tensor,
    dO: cute.Tensor,
    sum_OdO: cute.Tensor,
    lse: cute.Tensor,
    scaled_lse: cute.Tensor,
    cumulative_s_q: Union[cute.Tensor, None],
    sum_OdO_scale: Float32,
    lse_scale: Float32,
    problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
):
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, tidy, tidz = cute.arch.thread_idx()

    num_h_r = problem_shape[3][0][0]
    head_coord = ((bidy % num_h_r, bidy // num_h_r), bidz)

    seqlen_q = problem_shape[0]
    offset = 0
    if cutlass.const_expr(self.varlen):
        offset = cumulative_s_q[bidz]
        seqlen_q = cumulative_s_q[bidz + 1] - offset

    for idx_q_t in cutlass.range(tidy, self.sum_OdO_block_q, self.sum_OdO_num_threads_q, unroll_full=True):
        idx_q = idx_q_t + self.sum_OdO_block_q * bidx
        if idx_q < seqlen_q:
            O_bhq = O[idx_q + offset, None, head_coord]
            O_bhq = cute.logical_divide(O_bhq, cute.make_layout(self.sum_OdO_elem_per_load))
            dO_bhq = dO[idx_q + offset, None, head_coord]
            dO_bhq = cute.logical_divide(dO_bhq, cute.make_layout(self.sum_OdO_elem_per_load))

            idx_d_start = tidx
            idx_d_step = self.sum_OdO_num_threads_d
            acc = 0.0
            for idx_d in cutlass.range(idx_d_start, O.shape[1] // self.sum_OdO_elem_per_load, idx_d_step):
                O_frag = O_bhq[None, idx_d].load()
                dO_frag = dO_bhq[None, idx_d].load()
                prod_frag = O_frag * dO_frag
                prod_frag = prod_frag.to(self.acc_dtype)
                acc += prod_frag.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=0)

            acc = cute.arch.warp_reduction_sum(acc, threads_in_group=self.sum_OdO_num_threads_d)

            if tidx == 0:
                lse_bhq = lse[idx_q + offset, head_coord]
                sum_OdO[idx_q, head_coord] = sum_OdO_scale * acc
                scaled_lse[idx_q, head_coord] = lse_scale * lse_bhq


# =================== @cute.jit methods ===========================


@cute.jit
def quantize(input: cute.Tensor, frg_cnt: Int32, dtype) -> cute.Tensor:
    output = cute.make_rmem_tensor(input.shape, dtype)
    frg_tile = cute.size(input) // frg_cnt
    t_frg = cute.logical_divide(input, cute.make_layout(frg_cnt))
    output_frg = cute.make_tensor(output.iterator, t_frg.layout)
    for i in cutlass.range(frg_tile, unroll_full=True):
        frg_vec = t_frg[None, i].load()
        output_frg[None, i].store(frg_vec.to(dtype))
    return output


@cute.jit
def cvt_amax_to_e8m0_rp(amax: Float32):
    """Encode ``amax / 448`` as E8M0 RP without an FP32 divide.

    For a normal positive FP32 value, 448 is ``1.75 * 2**8``.  The
    upward-rounded scale exponent is therefore the amax exponent minus
    eight, plus one exactly when its significand is greater than 1.75.
    """
    amax_bits = amax.bitcast(cutlass.Uint32)
    exponent = (amax_bits >> 23) & cutlass.Uint32(0xFF)
    mantissa = amax_bits & cutlass.Uint32(0x7FFFFF)

    scale_exp = cutlass.Int32(exponent) - cutlass.Int32(8)
    if mantissa > cutlass.Uint32(0x600000):
        scale_exp = scale_exp + cutlass.Int32(1)
    if exponent == cutlass.Uint32(0xFF):
        scale_exp = cutlass.Int32(254)
    if scale_exp < cutlass.Int32(0):
        scale_exp = cutlass.Int32(0)
    if scale_exp > cutlass.Int32(254):
        scale_exp = cutlass.Int32(254)

    scale_bits = cutlass.Uint8(scale_exp)
    scale = scale_bits.bitcast(cutlass.Float8E8M0FNU)
    inv_scale = ((cutlass.Uint32(254) - cutlass.Uint32(scale_bits)) << 23).bitcast(cutlass.Float32)
    if scale_bits == cutlass.Uint8(0):
        inv_scale = cutlass.Float32(0.0)
    if scale_bits == cutlass.Uint8(254):
        inv_scale = cutlass.Uint32(1 << 22).bitcast(cutlass.Float32)
    return scale, inv_scale


@cute.jit
def store_constant_mxfp8_scales_to_tmem(
    self,
    tSF: cute.Tensor,
    tidx: Int32,
    scale_byte: int,
):
    """Fill an MMA scale tensor with one compile-time E8M0 byte."""
    tSF_compact = cute.filter_zeros(tSF)
    copy_atom_r2t = cute.make_copy_atom(
        tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(4)),
        self.sf_dtype,
    )
    tiled_copy_r2t = tcgen05.make_tmem_copy(copy_atom_r2t, tSF_compact)
    thr_copy_r2t = tiled_copy_r2t.get_slice(tidx)
    tRT_tSF = thr_copy_r2t.partition_D(tSF_compact)
    rSF = cute.make_rmem_tensor_like(tSF_compact, self.sf_dtype)
    tRT_rSF = thr_copy_r2t.partition_S(rSF)
    tRT_rSF_uint = cute.make_tensor(
        cute.recast_ptr(tRT_rSF.iterator, dtype=cutlass.Uint8),
        tRT_rSF.layout,
    )
    tRT_rSF_uint.fill(scale_byte)
    cute.copy(tiled_copy_r2t, tRT_rSF, tRT_tSF)


@cute.jit
def store_mxfp8_scales_to_tmem(
    scale_bytes: cute.Tensor,
    tSF: cute.Tensor,
    tidx: Int32,
):
    """Store packed E8M0 scale words directly to the SFA TMEM layout."""
    tSF_compact = cute.filter_zeros(tSF)
    # The four bytes for one row come from four warps across two CTAs.
    # tcgen05.st has no byte-granular form, so SMEM remains the required
    # cross-warp gather; its byte layout is already a packed Uint32 word.
    scale_words = cute.make_tensor(
        cute.recast_ptr(scale_bytes.iterator, dtype=cutlass.Uint32),
        cute.make_layout((64,), stride=(1,)),
    )
    row = tidx % 32
    packed_scales_lo = scale_words[row]
    packed_scales_hi = scale_words[row + 32]
    warp_rank = tidx // 32
    # One writer warp advances one TMEM column group; that address field
    # starts at bit 21 for this compact SFA layout.
    tmem_addr = Int32(tSF_compact.iterator.toint()) + (warp_rank << 21)
    tcgen05_st_32x32b_x4(
        tmem_addr,
        packed_scales_lo,
        packed_scales_hi,
    )


@cutlass.dsl_user_op
def tcgen05_st_32x32b_x4(
    tmem_addr: Int32,
    packed_scales_lo: Uint32,
    packed_scales_hi: Uint32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
):
    """Emit the scale-specific four-register TMEM store without an RMEM tensor."""
    vec_type = ir.VectorType.get([4], Uint32.mlir_type, loc=loc)
    packed_scales = vector.from_elements(
        vec_type,
        [
            packed_scales_lo.ir_value(loc=loc, ip=ip),
            packed_scales_hi.ir_value(loc=loc, ip=ip),
            packed_scales_lo.ir_value(loc=loc, ip=ip),
            packed_scales_hi.ir_value(loc=loc, ip=ip),
        ],
        loc=loc,
        ip=ip,
    )
    nvvm.tcgen05_st(
        nvvm.Tcgen05LdStShape.SHAPE_32X32B,
        llvm.inttoptr(
            llvm.PointerType.get(6),
            tmem_addr.ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        ),
        packed_scales,
        loc=loc,
        ip=ip,
    )


@cute.jit
def store(
    self,
    gmem: cute.Tensor,
    regs: cute.Tensor,
    coord: cute.Tensor,
    tensor_shape: cute.Shape,
):
    copy_atom_r2g = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        self.element_dtype,
        num_bits_per_copy=self.store_num_bits_per_copy,
    )
    tiled_copy_r2g = cute.make_cotiled_copy(
        copy_atom_r2g,
        cute.make_layout((1, self.store_num_bits_per_copy // self.element_dtype.width)),
        regs.layout,
    )
    thr_copy_r2g = tiled_copy_r2g.get_slice(0)

    tCg = thr_copy_r2g.partition_D(gmem)
    rmem_quant = quantize(regs, 4, self.element_dtype)
    tCr = thr_copy_r2g.partition_S(rmem_quant)
    tPc = thr_copy_r2g.partition_D(coord)

    # FIXME cute.copy expects mode 0 (atom_v,rest_v) to be removed
    #       Fix this so that the predicate tensor can simply be congruent to
    #       the original partitioned tensor
    preds_shape = (tPc.shape[0][1], tPc.shape[1], tPc.shape[2], tPc.shape[3])
    preds = cute.make_rmem_tensor(preds_shape, Boolean)
    for v in cutlass.range_constexpr(preds.shape[0]):
        for m in cutlass.range_constexpr(preds.shape[1]):
            for n in cutlass.range_constexpr(preds.shape[2]):
                for k in cutlass.range_constexpr(preds.shape[3]):
                    lhs = tPc[(0, v), m, n, k]
                    val = cute.elem_less(lhs, tensor_shape)
                    preds[v, m, n, k] = val
    cute.copy(copy_atom_r2g, tCr, tCg, pred=preds)


@cute.jit
def expand_last_SF_stride(
    sSF_layout: cute.Layout,
):
    """
    This function is used to expand the last stride of the SF tensor to 2x the original size for 2x64 s2t.
    """
    sSF_shape = sSF_layout.shape
    sSF_shape = (sSF_shape[0], sSF_shape[1], sSF_shape[2], sSF_shape[3], 2)
    sSF_stride = sSF_layout.stride
    sSF_stride = (sSF_stride[0], sSF_stride[1], sSF_stride[2], 2 * sSF_stride[3], sSF_stride[3])
    return cute.make_layout(sSF_shape, stride=sSF_stride)


def make_kq_mma_atoms(config, low_precision_type):
    """Create the common KQ block-scaled MMA atoms used by dK and dV.

    Keep this as a plain Python helper so construction remains in the caller's
    CuTe JIT trace.
    """
    tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
        low_precision_type,
        OperandMajorMode.K,
        OperandMajorMode.K,
        config.sf_dtype,
        config.sf_vec_size,
        config.cta_group,
        config.KQ_mma_tiler[:2],
        tcgen05.OperandSource.TMEM,
    )
    tiled_mma_smem = sm100_utils.make_blockscaled_trivial_tiled_mma(
        low_precision_type,
        OperandMajorMode.K,
        OperandMajorMode.K,
        config.sf_dtype,
        config.sf_vec_size,
        config.cta_group,
        config.KQ_mma_tiler[:2],
        tcgen05.OperandSource.SMEM,
    )
    tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
        low_precision_type,
        OperandMajorMode.K,
        OperandMajorMode.K,
        config.sf_dtype,
        config.sf_vec_size,
        tcgen05.CtaGroup.ONE,
        config.KQ_mma_tiler_sfb[:2],
        tcgen05.OperandSource.TMEM,
    )
    tiled_mma_sfa = sm100_utils.make_blockscaled_trivial_tiled_mma(
        low_precision_type,
        OperandMajorMode.K,
        OperandMajorMode.K,
        config.sf_dtype,
        config.sf_vec_size,
        config.cta_group,
        config.KQ_mma_tiler_sfa[:2],
        tcgen05.OperandSource.TMEM,
    )
    return tiled_mma, tiled_mma_smem, tiled_mma_sfb, tiled_mma_sfa


def make_kq_tma_setup(
    config,
    Q: cute.Tensor,
    K: cute.Tensor,
    SF_Q: cute.Tensor,
    SF_K: cute.Tensor,
    tiled_mma,
    tiled_mma_sfb,
    tiled_mma_sfa,
    load_mma_stage,
    low_precision_type,
    q_seq_extent=None,
    k_seq_extent=None,
):
    """Build common KQ SMEM layouts and Q/K/SFQ/SFK TMA descriptors.

    Keep this as a plain Python helper so all returned CuTe objects remain in
    the caller's JIT trace. For tight-packed varlen operands, Q/K carry the
    total token extent while their padded SF tensors use the per-batch maximum;
    pass those maxima through q_seq_extent and k_seq_extent.
    """
    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((*config.cluster_shape_mn, 1)),
        (tiled_mma.thr_id.shape,),
    )

    Q_smem_layout_staged = sm100_utils.make_smem_layout_b(
        tiled_mma,
        config.KQ_mma_tiler,
        low_precision_type,
        load_mma_stage,
    )
    K_smem_layout_staged = sm100_utils.make_smem_layout_a(
        tiled_mma,
        config.KQ_mma_tiler,
        low_precision_type,
        1,
    )
    sfQ_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
        tiled_mma_sfb,
        config.KQ_mma_tiler_sfq_load,
        config.sf_vec_size,
        config.SFQ_load_stage,
    )
    sfQ_smem_layout_staged = expand_last_SF_stride(sfQ_smem_layout_staged)
    sfK_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
        tiled_mma_sfb,
        config.KQ_mma_tiler_sfk_smem,
        config.sf_vec_size,
        config.SFK_load_stage,
    )
    sfK_smem_layout_staged = expand_last_SF_stride(sfK_smem_layout_staged)

    tma_load_op = cpasync.CopyBulkTensorTileG2SOp(config.cta_group)
    Q_smem_layout = cute.select(Q_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
        tma_load_op,
        Q,
        Q_smem_layout,
        config.KQ_mma_tiler,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )
    K_smem_layout = cute.select(K_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load_op,
        K,
        K_smem_layout,
        config.KQ_mma_tiler,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )

    q_seq_extent = Q.shape[0] if q_seq_extent is None else q_seq_extent
    k_seq_extent = K.shape[0] if k_seq_extent is None else k_seq_extent
    sfQ_shape = (
        cute.round_up(q_seq_extent, 128),
        Q.shape[1],
        cute.size(Q.shape[2]) * 2,
    )
    SF_Q = cute.make_tensor(
        SF_Q.iterator,
        blockscaled_utils.tile_atom_to_shape_SF(sfQ_shape, config.sf_vec_size),
    )
    sfK_shape = (
        cute.round_up(cute.size(k_seq_extent), 128) * 2,
        K.shape[1],
        cute.size(K.shape[2]) * 2,
    )
    SF_K = cute.make_tensor(
        SF_K.iterator,
        blockscaled_utils.tile_atom_to_shape_SF(sfK_shape, config.sf_vec_size),
    )

    sfa_op = cpasync.CopyBulkTensorTileG2SOp(config.cta_group)
    sfb_mcast_op = cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
    sfK_smem_layout = cute.slice_(sfK_smem_layout_staged, (None, None, None, 0, 0))
    tma_atom_sfK, tma_tensor_sfK = cute.nvgpu.make_tiled_tma_atom_A(
        sfa_op,
        SF_K,
        sfK_smem_layout,
        config.KQ_mma_tiler_sfk_load,
        tiled_mma_sfa,
        cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16,
    )
    sfQ_smem_layout = cute.slice_(sfQ_smem_layout_staged, (None, None, None, 0, 0))
    tma_atom_sfQ, tma_tensor_sfQ = cute.nvgpu.make_tiled_tma_atom_B(
        sfb_mcast_op,
        SF_Q,
        sfQ_smem_layout,
        config.KQ_mma_tiler_sfq_load,
        tiled_mma_sfb,
        cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16,
    )

    config.tma_copy_Q_bytes = cute.size_in_bytes(low_precision_type, Q_smem_layout)
    config.tma_copy_K_bytes = cute.size_in_bytes(low_precision_type, K_smem_layout)
    config.tma_copy_sfQ_bytes = cute.size_in_bytes(config.sf_dtype, sfQ_smem_layout)
    config.tma_copy_sfK_bytes = cute.size_in_bytes(config.sf_dtype, sfK_smem_layout)

    return (
        cluster_layout_vmnk,
        Q_smem_layout_staged,
        K_smem_layout_staged,
        sfQ_smem_layout_staged,
        sfK_smem_layout_staged,
        tma_load_op,
        sfa_op,
        sfb_mcast_op,
        tma_atom_Q,
        tma_tensor_Q,
        tma_atom_K,
        tma_tensor_K,
        tma_atom_sfQ,
        tma_tensor_sfQ,
        tma_atom_sfK,
        tma_tensor_sfK,
    )


@cute.jit
def mainloop_s2t_copy_and_partition_sf_2x64(self, sSF: cute.Tensor, tSF: cute.Tensor, is_SFA: Boolean):
    """S2T copy for SFA/SFB using Cp2x64x128b0213Op / Cp2x64x128b0123Op.

    IMPORTANT: sSF.shape[0][0][0] MUST be a flat integer (e.g. 32), not a tuple.
    Callers using hierarchical SF layouts must slice sSF and tSF to individual
    sub-groups before calling this function.
    """
    sSF_shape = sSF.shape
    sSF_stride = sSF.stride

    # Handle hierarchical mode-0-0-0:
    # (32, N) with strides (16, 4) → flatten N into the (2, 2) expansion:
    # (32, (2, 2*N)) = 32 rows × (2*2*N) elements/row
    _raw_s000 = sSF_shape[0][0][0]
    _raw_d000 = sSF_stride[0][0][0]
    _is_hier = type(_raw_s000) is tuple
    _s000 = _raw_s000[0] if _is_hier else _raw_s000  # 32
    _d000 = _raw_d000[0] if _is_hier else _raw_d000  # 16
    _n_sg = _raw_s000[1] if _is_hier else 1
    # Expansion: (2, 2*N) — multiply N into the sub-group count
    _exp_shape = (2, 2 * _n_sg)
    _exp_stride = (sSF_stride[4], 4)
    # Handle mode-0-0 — may be tuple or flat after reshape
    _s00 = sSF_shape[0][0]
    _d00 = sSF_stride[0][0]
    _s00_1 = _s00[1] if type(_s00) is tuple else 1
    _d00_1 = _d00[1] if type(_d00) is tuple else 0

    sSF_reshaped = cute.make_tensor(
        sSF.iterator,
        cute.make_layout(
            ((((_s000, _exp_shape), _s00_1), sSF_shape[0][1]), sSF_shape[1], sSF_shape[2], sSF_shape[3]),
            stride=((((_d000, _exp_stride), _d00_1), sSF_stride[0][1]), sSF_stride[1], sSF_stride[2], sSF_stride[3]),
        ),
    )

    tCsSF_compact = cute.filter_zeros(sSF_reshaped)
    tCtSF_compact = cute.filter_zeros(tSF)

    atom = tcgen05.Cp2x64x128b0213Op(self.cta_group) if is_SFA else tcgen05.Cp2x64x128b0123Op(self.cta_group)
    copy_atom_s2t = cute.make_copy_atom(
        atom,
        self.sf_dtype,
    )
    tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
    thr_copy_s2t = tiled_copy_s2t.get_slice(0)

    tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
    tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
    tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

    # Broadcast src mode-1 to match dst mode-1 for SFB sub-group copies.
    _src_m1 = cute.size(tCsSF_compact_s2t, mode=[1])
    _dst_m1 = cute.size(tCtSF_compact_s2t, mode=[1])
    _bc_m1 = _dst_m1 // _src_m1
    _ss = tCsSF_compact_s2t.shape
    _sd = tCsSF_compact_s2t.stride
    tCsSF_compact_s2t = cute.make_tensor(
        tCsSF_compact_s2t.iterator,
        cute.make_layout(
            (_ss[0], (_ss[1], _bc_m1), *_ss[2:]),
            stride=(_sd[0], (_sd[1], 0), *_sd[2:]),
        ),
    )

    return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t


@cute.jit
def mainloop_s2t_copy_and_partition_sfb_mn_2x64(
    self,
    sSF: cute.Tensor,
    tSF: cute.Tensor,
):
    """S2T copy for SFB when the 2-way hierarchy is on MN/N, not K."""
    sSF = sSF[None, None, None, None, 0]
    tCsSF_compact = cute.filter_zeros(sSF)
    tCtSF_compact = cute.filter_zeros(tSF)

    copy_atom_s2t = cute.make_copy_atom(
        tcgen05.Cp2x64x128b0123Op(self.cta_group),
        self.sf_dtype,
    )
    tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
    thr_copy_s2t = tiled_copy_s2t.get_slice(0)

    tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
    tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
    tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

    return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t


@cute.jit
def copy_grouped_sf_to_tmem(
    self,
    sSF: cute.Tensor,
    tSF: cute.Tensor,
    stage: Int32,
    is_SFA: Boolean,
):
    """Copy every compiler-static K-half of a grouped SFA/SFB tensor."""
    k_halves = cute.size(tSF, mode=[3])
    for k_half in cutlass.range_constexpr(k_halves):
        tiled_copy, tCsSF, tCtSF = mainloop_s2t_copy_and_partition_sf_2x64(
            self,
            sSF,
            tSF[(None, None, None, k_half)],
            is_SFA=is_SFA,
        )
        cute.copy(
            tiled_copy,
            tCsSF[None, None, None, None, stage * k_halves + k_half],
            tCtSF,
        )


@cute.jit
def copy_grouped_sfb_mn_to_tmem(
    self,
    sSF: cute.Tensor,
    tSF: cute.Tensor,
    stage: Int32,
):
    """Copy every K-half of grouped SFB with its hierarchy on MN/N."""
    k_halves = cute.size(tSF, mode=[3])
    for k_half in cutlass.range_constexpr(k_halves):
        tiled_copy, tCsSF, tCtSF = mainloop_s2t_copy_and_partition_sfb_mn_2x64(
            self,
            sSF,
            tSF[(None, None, None, k_half)],
        )
        cute.copy(
            tiled_copy,
            tCsSF[None, None, None, None, stage * k_halves + k_half],
            tCtSF,
        )


# =================== @staticmethod methods
# =================== @staticmethod methods =======================


def compute_grid(
    output_shape: cute.Shape,
    cta_tiler: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """Return the non-persistent CTA grid for an (M, D, ((Hr, Hk), B)) tensor."""
    return (
        cute.ceil_div(cute.size(output_shape[0]), cta_tiler[0]),
        cute.size(output_shape[2][0]),
        cute.size(output_shape[2][1]),
    )


def compute_sum_odo_grid(
    problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
    block_q: int,
) -> Tuple[int, int, int]:
    grid = (
        cute.ceil_div(cute.size(problem_shape[0]), block_q),
        cute.size(problem_shape[3][0]),  # H
        cute.size(problem_shape[3][1]),  # B
    )
    return grid


# =================== Parameterized shared methods ================


def get_workspace_tensor(
    self,
    problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
    workspace: cute.Tensor,
    acc_dtype: Type[cutlass.Numeric],
    needs_dq_acc: bool = True,
) -> Tuple[cute.Tensor, cute.Tensor, cute.Tensor]:
    Q, D = problem_shape[0], problem_shape[2]
    H, B = cute.size(problem_shape[3][0]), cute.size(problem_shape[3][1])
    H_r, H_k = problem_shape[3][0]
    D = cute.round_up(D, 8)
    Q = cute.round_up(Q, 8)

    acc_bytes = acc_dtype.width // 8
    sum_OdO_bytes = cute.assume(B * H * Q * acc_bytes, divby=acc_bytes)
    scaled_lse_bytes = cute.assume(B * H * Q * acc_bytes, divby=acc_bytes)

    sum_OdO_iter = workspace.iterator
    scaled_lse_iter = sum_OdO_iter + sum_OdO_bytes

    if needs_dq_acc:
        dQ_acc_iter = scaled_lse_iter + scaled_lse_bytes

    sum_OdO_iter = cute.recast_ptr(sum_OdO_iter, dtype=self.acc_dtype)
    scaled_lse_iter = cute.recast_ptr(scaled_lse_iter, dtype=self.acc_dtype)

    sum_OdO = cute.make_tensor(
        sum_OdO_iter,
        cute.make_layout((Q, ((H_r, H_k), B)), stride=(1, ((Q, Q * H_r), Q * H))),
    )
    scaled_lse = cute.make_tensor(
        scaled_lse_iter,
        cute.make_layout((Q, ((H_r, H_k), B)), stride=(1, ((Q, Q * H_r), Q * H))),
    )

    if needs_dq_acc:
        dQ_acc_iter = cute.recast_ptr(dQ_acc_iter, dtype=self.acc_dtype)
        dQ_acc = cute.make_tensor(
            dQ_acc_iter,
            cute.make_layout(
                (Q, D, ((H_r, H_k), B)),
                stride=(D, 1, ((D * Q, D * Q * H_r), D * Q * H)),
            ),
        )
    else:
        dQ_acc = None

    return sum_OdO, scaled_lse, dQ_acc


def get_workspace_size(
    q: int,
    d: int,
    h: int,
    b: int,
    acc_dtype: Type[cutlass.Numeric],
    needs_dq_acc: bool = True,
):
    d = (d + 7) // 8 * 8  # round up to 8
    q = (q + 7) // 8 * 8  # round up to 8
    workspace_bytes = 0
    # OdO vector
    workspace_bytes += b * h * q * acc_dtype.width // 8
    # scaled LSE vector
    workspace_bytes += b * h * q * acc_dtype.width // 8
    if needs_dq_acc:
        # FP32 versions of outputs that are churned (start off with Q only)
        workspace_bytes += b * h * q * d * acc_dtype.width // 8
    return workspace_bytes


def epilogue_tmem_copy_and_partition(
    copy_atom_t2r,
    tmem_tensor,
    coord_tensor,
    gmem_tensor,
    thread_idx,
    acc_dtype,
):
    """Build one epilogue TMEM-to-register copy mapping.

    Keep the construction order aligned with the original kernel-local code:
    tiled copy, thread slice, coordinate/global destinations, register
    fragment, then TMEM source. The helper only creates views and fragments;
    the caller retains pipeline synchronization, the actual copy, conversion,
    predication, and global store.
    """
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tmem_tensor)
    thr_copy_t2r = tiled_copy_t2r.get_slice(thread_idx)
    tTR_coord = thr_copy_t2r.partition_D(coord_tensor)
    tTR_gmem = thr_copy_t2r.partition_D(gmem_tensor)
    tTR_rmem = cute.make_rmem_tensor(tTR_coord.shape, acc_dtype)
    tTR_tmem = thr_copy_t2r.partition_S(tmem_tensor)
    return tiled_copy_t2r, tTR_tmem, tTR_rmem, tTR_gmem, tTR_coord


def make_tma_umma_pipeline(
    mbar_ptr,
    num_stages,
    tx_count,
    cluster_layout_vmnk,
    producer_threads,
    consumer_threads,
):
    """Construct a TMA-producer/UMMA-consumer pipeline."""
    producer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        producer_threads,
    )
    consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        consumer_threads,
    )
    return pipeline.PipelineTmaUmma.create(
        barrier_storage=mbar_ptr,
        num_stages=num_stages,
        producer_group=producer_group,
        consumer_group=consumer_group,
        tx_count=tx_count,
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    )


def make_umma_async_pipeline(
    mbar_ptr,
    num_stages,
    cluster_layout_vmnk,
    producer_threads,
    consumer_threads,
):
    """Construct a UMMA-producer/async-consumer pipeline."""
    producer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        producer_threads,
    )
    consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        consumer_threads,
    )
    return pipeline.PipelineUmmaAsync.create(
        barrier_storage=mbar_ptr,
        num_stages=num_stages,
        producer_group=producer_group,
        consumer_group=consumer_group,
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    )


def make_pipeline_umma_async(self, mbar_ptr, num_stages, cluster_layout_vmnk):
    """Construct the codegen-sensitive dQ compute-0 pipeline.

    Public CuTe 4.6.x requires this specialized helper boundary for dQ causal
    GQA. Keep it separate from the fully parameterized pipeline factories.
    """
    producer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        len([self.mma_warp_id]),
    )
    consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        self.num_compute_0_warps * self.threads_per_warp * cute.size(cluster_layout_vmnk, mode=[0]),
    )
    return pipeline.PipelineUmmaAsync.create(
        barrier_storage=mbar_ptr,
        num_stages=num_stages,
        producer_group=producer_group,
        consumer_group=consumer_group,
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    )


def make_async_umma_pipeline(
    mbar_ptr,
    num_stages,
    cluster_layout_vmnk,
    producer_threads,
    consumer_threads,
):
    """Construct an async-producer/UMMA-consumer pipeline."""
    producer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        producer_threads,
    )
    consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        consumer_threads,
    )
    return pipeline.PipelineAsyncUmma.create(
        barrier_storage=mbar_ptr,
        num_stages=num_stages,
        producer_group=producer_group,
        consumer_group=consumer_group,
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    )


@cute.jit
def split_wg(
    t: cute.Tensor,
    num_warp_groups: Int32,
    wg_idx: Int32,
) -> cute.Tensor:
    """Split the last layout mode across warp groups for ranks one through four."""
    ret = None
    rank = cute.rank(t.layout)
    last_mode = rank - 1
    mode_size = cute.size(t, mode=[last_mode])
    split_shape = (num_warp_groups, mode_size // num_warp_groups)
    split_coord = (wg_idx, None)
    if cutlass.const_expr(rank == 1):
        p = cute.composition(t, cute.make_layout((split_shape,)))
        ret = p[split_coord]
    elif cutlass.const_expr(rank == 2):
        p = cute.composition(t, cute.make_layout((t.shape[0], split_shape)))
        ret = p[None, split_coord]
    elif cutlass.const_expr(rank == 3):
        p = cute.composition(t, cute.make_layout((t.shape[0], t.shape[1], split_shape)))
        ret = p[None, None, split_coord]
    else:
        p = cute.composition(
            t,
            cute.make_layout((t.shape[0], t.shape[1], t.shape[2], split_shape)),
        )
        ret = p[None, None, None, split_coord]
    return ret


@cute.jit
def split_wg_interleaved(t: cute.Tensor, num_warp_groups: Int32, wg_idx: Int32) -> cute.Tensor:
    """Baseline-equivalent interleaved split of the final tensor mode."""
    ret = None
    if cutlass.const_expr(cute.rank(t.layout) == 1):
        p = cute.composition(t, cute.make_layout(((cute.size(t) // num_warp_groups, num_warp_groups),)))
        ret = p[(None, wg_idx)]
    elif cutlass.const_expr(cute.rank(t.layout) == 2):
        p = cute.composition(
            t,
            cute.make_layout((t.shape[0], (cute.size(t, mode=[1]) // num_warp_groups, num_warp_groups))),
        )
        ret = p[None, (None, wg_idx)]
    elif cutlass.const_expr(cute.rank(t.layout) == 3):
        p = cute.composition(
            t,
            cute.make_layout(
                (
                    t.shape[0],
                    t.shape[1],
                    (cute.size(t, mode=[2]) // num_warp_groups, num_warp_groups),
                )
            ),
        )
        ret = p[None, None, (None, wg_idx)]
    else:
        p = cute.composition(
            t,
            cute.make_layout(
                (
                    t.shape[0],
                    t.shape[1],
                    t.shape[2],
                    (cute.size(t, mode=[3]) // num_warp_groups, num_warp_groups),
                )
            ),
        )
        ret = p[None, None, None, (None, wg_idx)]
    return ret


@cute.jit
def split_wg_contiguous(t: cute.Tensor, num_warp_groups: Int32, wg_idx: Int32) -> cute.Tensor:
    return split_wg(t, num_warp_groups, wg_idx)


# --- D256 primitives (fmha_cute/d256_primitives.py) ---------------------


@cute.jit
def copy_mxfp8_scale_tile_to_tmem(
    scale_bytes: cute.Tensor,
    tSF: cute.Tensor,
):
    """Copy one 32x16-byte SFA tile to four TMEM warp replicas via tcgen05.cp."""
    tSF_compact = cute.filter_zeros(tSF)
    smem_desc = nvvm.tcgen05_mma_smem_desc(
        Int32(scale_bytes.iterator.toint() >> 4).ir_value(),
        Int32(1).ir_value(),
        Int32(8).ir_value(),
        Int8(0).ir_value(),
        Boolean(False).ir_value(),
        Int8(0).ir_value(),
    )
    tmem_ptr = llvm.inttoptr(
        llvm.PointerType.get(6),
        Int32(tSF_compact.iterator.toint()).ir_value(),
    )
    nvvm.tcgen05_cp(
        nvvm.Tcgen05CpShape.SHAPE_32x128b,
        tmem_ptr,
        smem_desc,
        group=nvvm.CTAGroupKind.CTA_2,
        multicast=nvvm.Tcgen05CpMulticast.WARPX4,
    )


@cute.jit
def store_identity_mxfp8_scales_to_tmem(
    config,
    tSF: cute.Tensor,
    tidx: Int32,
):
    """Fill a D256 fixed-scale TMEM tensor with E8M0 byte 127 (scale 1)."""
    store_constant_mxfp8_scales_to_tmem(config, tSF, tidx, 127)


def make_clc_fetch_pipeline(
    mbar_ptr,
    num_stages,
    cluster_layout_vmnk,
    consumer_threads,
    tx_count,
):
    """Construct the CLC scheduler fetch pipeline for persistent D256."""
    producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        consumer_threads,
    )
    return pipeline.PipelineClcFetchAsync.create(
        barrier_storage=mbar_ptr,
        num_stages=num_stages,
        producer_group=producer_group,
        consumer_group=consumer_group,
        tx_count=tx_count,
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    )
