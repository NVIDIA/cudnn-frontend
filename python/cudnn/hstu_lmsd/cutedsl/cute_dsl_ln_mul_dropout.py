# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL implementation of the ln_mul_dropout forward.

Fuses the Triton mask-generation and layer-norm kernels into one kernel.
Enabled with HSTU_CUTE_LN=1.

Four changes over the previous version, none of which alters the arithmetic:

  TV_ROW       an output tile is one row rather than ROWS rows, so the row index
               comes from the loop instead of thread_idx // 32. Drops an address
               computation and 8 registers (80 -> 72), lifting warps_active from
               71% to 83%.
  PACK_TAIL    the tail evaluates element pairs with fma/mul/add/sub_packed_f32x2.
               Accumulation stays FP32, so the results are unchanged.
               exp_packed_f32x2 is deliberately not used: it is a natural
               exponential, not exp2, and measured slower than two scalar exp2.
  REREAD_X     x is read twice -- once for the row sum, once to normalise --
               and was previously held in registers across both. Dropping it
               after the reduction and re-reading in the compute pass trades an
               extra read for a shorter live range, and the row is still in L1
               from moments earlier. Worth 2.5% at TPR=32; it costs 2.1% at
               TPR=16, so it is tied to the default configuration.
  PHILOX32     full 32-bit dropout samples. 16-bit halves let one philox4x32
               cover eight elements instead of four, but quantise the keep rate
               to 1/65536: at p=0.1 the drop rate is 0.10000610 rather than
               0.100000005. That bias is below the sampling noise of a single
               step (8.0e-6 over N*D draws) and 576x below one bf16 ULP, so it
               is not observable -- but it costs 31% to remove and the decision
               was to remove it.
  I64_REBASE   moves every row-bearing input/output to the current row with
               64-bit pointer arithmetic. Subsequent CuTe indexing is local to
               that row, so large shapes can run in one launch without
               overflowing a 32-bit byte offset.
  TPR          threads per row. TPR=32 is the default and keeps every output
               bit-identical to the previous kernel. TPR=16 puts two rows on a
               warp and shortens the reduction butterfly from five levels to
               four; the arithmetic is the same but the row sum is accumulated in
               a different order. Only the mask stays bit-identical then -- mean,
               rstd and y all shift by a few ULP, y because it is computed from
               mean and rstd. ~6% faster than the default, off unless
               the caller can accept that.

Measured on Rubin sm_107a, 212 SMs, memory clock locked to 4752 MHz, at
N=2,739,421 x D=512 bf16, against the previous CuTe DSL forward:

    previous                              1.000x   (1.824 ms)
    TPR=32, ROWS=2, REREAD_X (default)    0.933x   every output bit-identical
    TPR=16, ROWS=1 (opt-in)               0.899x   only the mask is identical

Ratios rather than milliseconds: the two hecate partitions are not equivalent
even under --constraint=cr (the same shipping kernel measures 1.825 ms on
batch-xdr and 1.932 ms on batch-spx), so absolute times only compare within one
job. Every test prints the previous kernel's time alongside for that reason.

ROWS is tied to TPR and has to be swept with it: ROWS=1 is right for TPR=16 but
gives no speed-up at all at TPR=32 (1.001x), where ROWS=2 wins.

The operator's measured speed-of-light (same-stream zero-arithmetic probe) is
1.183 ms. The remaining gap is instruction issue, not bandwidth: DRAM sits at
38% while issue_active is 72%, and 44% of thread-instructions are integer work
that is almost entirely Philox. Directions that were measured and rejected are
listed at the bottom.
"""

import math
import os

from cuda.bindings import driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as _cm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from cudnn.datatypes import _convert_to_cutlass_data_type

VEC = 8
ROWS = 2
ALIGN = 16
MIN_BLOCKS_PER_MP = 0

LOG2E = 1.4426950408889634

HOIST = True
DIV_MODE = "approx"
TV_ROW = True  # output tile = one row
# 32 keeps every output bit-identical; 16 is ~4% faster and moves
# mean/rstd by ~2 ULP. Must be a power of two: the reduction butterfly
# relies on 1 << off staying inside the aligned lane subgroup.
TPR = 32
# Full-tile requirement for low-level calls: D % (TPR * VEC) == 0.
PACK_TAIL = True  # f32x2 arithmetic in the tail
REREAD_X = True  # drop x after the reduction, load it again
M0, M1 = 0xD2511F53, 0xCD9E8D57
W0, W1 = 0x9E3779B9, 0xBB67AE85
MASK32 = 0xFFFFFFFF
ROUNDS = 10
# 16-bit halves give a 1.5e-5 quantisation error in the keep rate and let one
# philox4x32 cover 8 elements. Full 32-bit words drop the error to ~5e-9 (the
# residual is float32(p) itself, which Triton also has) at the cost of one
# Philox call per 4 elements instead of 8.
PHILOX32 = True


@dsl_user_op
def _domain_offset_i64(
    coord: cute.Coord,
    tensor: cute.Tensor,
    *,
    loc=None,
    ip=None,
) -> cute.Tensor:
    """Rebase a tensor with a 64-bit byte offset, preserving its layout."""
    flat_coord_i64 = tuple(cutlass.Int64(c) for c in cute.flatten(coord))
    flat_stride = cute.flatten_to_tuple(tensor.stride)
    assert len(flat_coord_i64) == len(flat_stride)
    offset = sum(c * s for c, s in zip(flat_coord_i64, flat_stride))
    assert isinstance(tensor.iterator, cute.Pointer)
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


def _keep_threshold32(p: float) -> int:
    """Smallest w in [0, 2**32] with float32(w * 2**-32) > float32(p).

    float32(w) is monotone in w so a binary search is exact; looping over the
    whole range the way the 16-bit version does is not an option at this width.
    """
    import struct

    def f32(x):
        return struct.unpack("f", struct.pack("f", x))[0]

    p32 = f32(p)
    scale = f32(2.0**-32)
    lo, hi = 0, 1 << 32
    while lo < hi:
        mid = (lo + hi) // 2
        if f32(f32(mid) * scale) > p32:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _keep_threshold(p: float) -> int:
    """Make the dropout decision in the integer domain: keep <=> h*(1/65536) > p <=> h >= T.

    1/65536 is a power of two, so the multiplication is exact and this is
    bit-equivalent to the floating-point decision, while saving a u32->f32
    conversion, an f32 multiply and an f32 compare -- 24 times per thread per tile.
    """
    import struct

    p32 = struct.unpack("f", struct.pack("f", p))[0]
    inv = 1.0 / 65536.0
    for h in range(65537):
        if struct.unpack("f", struct.pack("f", h * inv))[0] > p32:
            return h
    return 65536


class LnMulDropoutForward:
    """Compile-time configuration and device code for LMSD forward.

    The class follows QuACK's split between a JIT-callable launch layer and a
    ``kernel`` method.  Performance-sensitive choices remain compile-time
    attributes, while the PyTorch wrapper only owns allocation and dispatch.
    Tensor names use the same scope convention: ``m`` for whole tensors,
    ``g`` for tiled global-memory views, ``t`` for per-thread partitions, and
    ``r`` for register fragments.
    """

    def __init__(self):
        self.threads_per_row = TPR
        self.rows_per_cta = ROWS
        self.vector_size = VEC
        self.tile_one_row = TV_ROW
        self.pack_tail = PACK_TAIL
        self.reread_x = REREAD_X
        self.philox32 = PHILOX32
        self.div_mode = DIV_MODE
        self.hoist = HOIST
        self.min_blocks_per_mp = MIN_BLOCKS_PER_MP

    @cute.kernel
    def kernel(
        self,
        gX: cute.Tensor,
        gU: cute.Tensor,
        gW: cute.Tensor,
        gB: cute.Tensor,
        gY0: cute.Tensor,
        gY1: cute.Tensor,
        gY2: cute.Tensor,
        gM: cute.Tensor,
        gMean: cute.Tensor,
        gRstd: cute.Tensor,
        eps: cutlass.Float32,
        drop: cutlass.Float32,
        thresh: cutlass.Uint32,
        thr_layout: cute.Layout,
        val_layout: cute.Layout,
        NTILE: cutlass.Constexpr,
        seed: cutlass.Int64,
        nrows: cutlass.Int32,
        row_off: cutlass.Int32,
        ncols: cutlass.Int32,
    ):
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        # A CTA owns rows_per_cta warps. Reducing threads_per_row below 32 lets
        # one warp process multiple rows without changing the physical block.
        thread_in_row = thread_idx % self.threads_per_row
        rows_per_block = self.rows_per_cta * 32 // self.threads_per_row
        row = block_idx * rows_per_block + thread_idx // self.threads_per_row
        valid = row < nrows
        if valid:
            tensor_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type)
            mask_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gM.element_type)
            # A one-row tile is addressed by the thread's position within its
            # row; a multi-row tile is addressed by the CTA thread id.
            copy_thread_idx = thread_in_row if cutlass.const_expr(self.tile_one_row) else thread_idx
            thread_copy = cute.make_tiled_copy_tv(tensor_copy_atom, thr_layout, val_layout).get_slice(copy_thread_idx)
            mask_thread_copy = cute.make_tiled_copy_tv(mask_copy_atom, thr_layout, val_layout).get_slice(copy_thread_idx)

            tile_row = row if cutlass.const_expr(self.tile_one_row) else block_idx
            row_coord = ((0, 0), (tile_row, 0))
            gX_row = _domain_offset_i64(row_coord, gX)
            gU_row = _domain_offset_i64(row_coord, gU)
            gY0_row = _domain_offset_i64(row_coord, gY0)
            gY1_row = _domain_offset_i64(row_coord, gY1)
            gY2_row = _domain_offset_i64(row_coord, gY2)
            gM_row = _domain_offset_i64(row_coord, gM)

            rX_tiles, rU_tiles, tXgY_tiles, tXgMask_tiles = [], [], [], []
            tXgX_tiles, tXgW_tiles, tXgB_tiles = [], [], []
            for j in cutlass.range_constexpr(NTILE):
                row_tile_coord = ((None, None), (0, j))
                for src, src_coord, dst in (
                    (gX_row, row_tile_coord, rX_tiles),
                    (gU_row, row_tile_coord, rU_tiles),
                ):
                    t = thread_copy.partition_S(src[src_coord])
                    f = cute.make_fragment_like(t)
                    cute.copy(tensor_copy_atom, t, f)
                    dst.append(f)
                if cutlass.const_expr(self.reread_x):
                    # keep the partition so the compute pass can fetch x again; the
                    # fragment itself dies with the reduction below
                    tXgX_tiles.append(thread_copy.partition_S(gX_row[row_tile_coord]))
                param_coord = ((None, None), (0, j))
                tXgW_tiles.append(thread_copy.partition_S(gW[param_coord]))
                tXgB_tiles.append(thread_copy.partition_S(gB[param_coord]))
                tXgY_tiles.append(
                    (
                        thread_copy.partition_S(gY0_row[row_tile_coord]),
                        thread_copy.partition_S(gY1_row[row_tile_coord]),
                        thread_copy.partition_S(gY2_row[row_tile_coord]),
                    )
                )
                tXgMask_tiles.append(mask_thread_copy.partition_S(gM_row[row_tile_coord]))

            sum_x = cutlass.Float32(0.0)
            sum_sq_x = cutlass.Float32(0.0)
            for j in cutlass.range_constexpr(NTILE):
                for e in cutlass.range_constexpr(self.vector_size):
                    v = rX_tiles[j][e].to(cutlass.Float32)
                    sum_x = sum_x + v
                    sum_sq_x = sum_sq_x + v * v
            # The butterfly stays inside the aligned threads-per-row subgroup.
            for off in cutlass.range_constexpr(self.threads_per_row.bit_length() - 1):
                sum_x = sum_x + cute.arch.shuffle_sync_bfly(sum_x, 1 << off)
                sum_sq_x = sum_sq_x + cute.arch.shuffle_sync_bfly(sum_sq_x, 1 << off)

            inv_d = cutlass.Float32(1.0) / ncols.to(cutlass.Float32)
            mean = sum_x * inv_d
            var = _cm.max(sum_sq_x * inv_d - mean * mean, cutlass.Float32(0.0))
            rstd = cutlass.Float32(1.0) / _cm.sqrt(var + eps)
            scale = cutlass.Float32(1.0) / (cutlass.Float32(1.0) - drop)

            k0 = cutlass.Uint32(seed & MASK32)
            k1 = cutlass.Uint32((seed >> 32) & MASK32)

            for j in cutlass.range_constexpr(NTILE):
                tXgW = tXgW_tiles[j]
                tXgB = tXgB_tiles[j]
                tXgY = tXgY_tiles[j]
                tXgMask = tXgMask_tiles[j]
                rU = rU_tiles[j]
                if cutlass.const_expr(self.reread_x):
                    # the row was read microseconds ago by this same CTA, so this
                    # should be an L1/L2 hit rather than a DRAM trip -- whether it
                    # actually is decides the whole experiment
                    rX = cute.make_fragment_like(tXgX_tiles[j])
                    cute.copy(tensor_copy_atom, tXgX_tiles[j], rX)
                else:
                    rX = rX_tiles[j]
                rW = cute.make_fragment_like(tXgW)
                rB = cute.make_fragment_like(tXgB)
                cute.copy(tensor_copy_atom, tXgW, rW)
                cute.copy(tensor_copy_atom, tXgB, rB)
                rY = cute.make_fragment_like(tXgY[0])
                rXOut = cute.make_fragment_like(tXgY[1])
                rUOut = cute.make_fragment_like(tXgY[2])
                rMask = cute.make_fragment_like(tXgMask)
                # A thread owns vector_size contiguous columns in tile j.
                # 16-bit: one call per (stream, 8 elements); the counter
                #         indexes the block of 8 columns.
                # 32-bit: one call per (stream, 4 elements); the counter indexes
                #         the block of 4, i.e. 2*philox_block + half.
                philox_block = cutlass.Uint32(j * self.threads_per_row) + cutlass.Uint32(thread_in_row)
                philox_words = []
                for mask_plane in cutlass.range_constexpr(3):
                    for h in cutlass.range_constexpr(2 if self.philox32 else 1):
                        c0 = cutlass.Uint32(row) + cutlass.Uint32(row_off)
                        c1 = philox_block * cutlass.Uint32(2) + cutlass.Uint32(h) if cutlass.const_expr(self.philox32) else philox_block
                        c2 = cutlass.Uint32(mask_plane)
                        c3 = cutlass.Uint32(0)
                        kk0, kk1 = k0, k1
                        m0c = cutlass.Uint32(M0)
                        m1c = cutlass.Uint32(M1)
                        for _r in cutlass.range_constexpr(ROUNDS):
                            # One mul.wide.u32 yields both halves of each
                            # product, so no second multiply is required.
                            p0 = cute.arch.mul_wide(m0c, c0)
                            p1 = cute.arch.mul_wide(m1c, c2)
                            hi0 = (p0 >> cutlass.Uint64(32)).to(cutlass.Uint32)
                            lo0 = (p0 & cutlass.Uint64(MASK32)).to(cutlass.Uint32)
                            hi1 = (p1 >> cutlass.Uint64(32)).to(cutlass.Uint32)
                            lo1 = (p1 & cutlass.Uint64(MASK32)).to(cutlass.Uint32)
                            c0 = hi1 ^ c1 ^ kk0
                            c1 = lo1
                            c2 = hi0 ^ c3 ^ kk1
                            c3 = lo0
                            kk0 = kk0 + cutlass.Uint32(W0)
                            kk1 = kk1 + cutlass.Uint32(W1)
                        philox_words.append((c0, c1, c2, c3))
                for e in cutlass.range_constexpr(self.vector_size):
                    bits = cutlass.Int8(0)
                    for mask_plane in cutlass.range_constexpr(3):
                        if cutlass.const_expr(self.philox32):
                            word = philox_words[mask_plane * 2 + e // 4][e % 4]
                            keep = word >= thresh
                        else:
                            word = philox_words[mask_plane][e // 2]
                            half = (word >> cutlass.Uint32(16 * (e % 2))) & cutlass.Uint32(0xFFFF)
                            keep = half >= thresh
                        bits = bits | (cutlass.Int8(1 << mask_plane) if keep else cutlass.Int8(0))
                    rMask[e] = bits
                if cutlass.const_expr(self.pack_tail):
                    # e and e+1 are independent; accumulation stays FP32 so the
                    # outputs are unchanged. There is no packed divide, and
                    # exp_packed_f32x2 measured slower than two scalar exp2.
                    one2 = (cutlass.Float32(1.0), cutlass.Float32(1.0))
                    mean2 = (mean, mean)
                    rstd2 = (rstd, rstd)
                    scale2 = (scale, scale)
                    nl2 = cutlass.Float32(-LOG2E)
                    for p in cutlass.range_constexpr(self.vector_size // 2):
                        e0 = 2 * p
                        e1 = e0 + 1
                        xf2 = (rX[e0].to(cutlass.Float32), rX[e1].to(cutlass.Float32))
                        uf2 = (rU[e0].to(cutlass.Float32), rU[e1].to(cutlass.Float32))
                        wv2 = (rW[e0].to(cutlass.Float32), rW[e1].to(cutlass.Float32))
                        bv2 = (rB[e0].to(cutlass.Float32), rB[e1].to(cutlass.Float32))
                        xh2 = cute.arch.mul_packed_f32x2(cute.arch.sub_packed_f32x2(xf2, mean2), rstd2)
                        ln2 = cute.arch.fma_packed_f32x2(xh2, wv2, bv2)
                        ex2 = (cute.arch.exp2(uf2[0] * nl2), cute.arch.exp2(uf2[1] * nl2))
                        den2 = cute.arch.add_packed_f32x2(ex2, one2)
                        su2 = (_cm.div(uf2[0], den2[0], approx=True), _cm.div(uf2[1], den2[1], approx=True))
                        sus2 = cute.arch.mul_packed_f32x2(su2, scale2)
                        xs2 = cute.arch.mul_packed_f32x2(xf2, scale2)
                        ys2 = cute.arch.mul_packed_f32x2(ln2, sus2)
                        zero = cutlass.Float32(0.0)
                        for q in cutlass.range_constexpr(2):
                            mb = rMask[e0 + q].to(cutlass.Int32)
                            rUOut[e0 + q] = (sus2[q] if (mb & 4) != 0 else zero).to(gX.element_type)
                            rXOut[e0 + q] = (xs2[q] if (mb & 2) != 0 else zero).to(gX.element_type)
                            rY[e0 + q] = (ys2[q] if (mb & 1) != 0 else zero).to(gX.element_type)
                for e in cutlass.range_constexpr(0 if self.pack_tail else self.vector_size):
                    xf = rX[e].to(cutlass.Float32)
                    uf = rU[e].to(cutlass.Float32)
                    wv = rW[e].to(cutlass.Float32)
                    bv = rB[e].to(cutlass.Float32)
                    ln = (xf - mean) * rstd * wv + bv
                    den = cutlass.Float32(1.0) + cute.arch.exp2(-uf * cutlass.Float32(LOG2E))
                    if cutlass.const_expr(self.div_mode == "approx"):
                        su = _cm.div(uf, den, approx=True)
                    elif cutlass.const_expr(self.div_mode == "rcp"):
                        su = uf * cute.arch.rcp_approx(den)
                    else:
                        su = uf / den
                    mb = rMask[e].to(cutlass.Int32)
                    zero = cutlass.Float32(0.0)
                    if cutlass.const_expr(self.hoist):
                        su_s = su * scale
                        rUOut[e] = (su_s if (mb & 4) != 0 else zero).to(gX.element_type)
                        rXOut[e] = (xf * scale if (mb & 2) != 0 else zero).to(gX.element_type)
                        rY[e] = (ln * su_s if (mb & 1) != 0 else zero).to(gX.element_type)
                    else:
                        rUOut[e] = (su * scale if (mb & 4) != 0 else zero).to(gX.element_type)
                        rXOut[e] = (xf * scale if (mb & 2) != 0 else zero).to(gX.element_type)
                        rY[e] = (ln * su * scale if (mb & 1) != 0 else zero).to(gX.element_type)
                cute.copy(tensor_copy_atom, rUOut, tXgY[0])
                cute.copy(tensor_copy_atom, rXOut, tXgY[1])
                cute.copy(tensor_copy_atom, rY, tXgY[2])
                cute.copy(mask_copy_atom, rMask, tXgMask)

            if thread_in_row == 0:
                gMean[row] = mean
                gRstd[row] = rstd

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mU: cute.Tensor,
        mW: cute.Tensor,
        mB: cute.Tensor,
        mY0: cute.Tensor,
        mY1: cute.Tensor,
        mY2: cute.Tensor,
        mM: cute.Tensor,
        mMean: cute.Tensor,
        mRstd: cute.Tensor,
        seed: cutlass.Int64,
        nrows: cutlass.Int32,
        row_off: cutlass.Int32,
        eps: cutlass.Float32,
        drop: cutlass.Float32,
        thresh: cutlass.Uint32,
        ncols: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        thr_layout = cute.make_ordered_layout(
            (1, self.threads_per_row) if self.tile_one_row else (self.rows_per_cta, 32),
            order=(1, 0),
        )
        # The CTA still has rows_per_cta warps when the output tile is one row,
        # so derive the grid from the physical row ownership, not thr_layout.
        row_thr_layout = cute.make_ordered_layout(
            (self.rows_per_cta * 32 // self.threads_per_row, 32),
            order=(1, 0),
        )
        val_layout = cute.make_ordered_layout((1, self.vector_size), order=(1, 0))
        _row_tiler, _ = cute.make_layout_tv(row_thr_layout, val_layout)
        grid_n = cute.size(cute.zipped_divide(mX, _row_tiler), mode=[1, 0])
        tiler, _ = cute.make_layout_tv(thr_layout, val_layout)
        tile = lambda tensor: cute.zipped_divide(tensor, tiler)
        gX = tile(mX)
        # Weight and bias are public rank-1 tensors. The device kernel only
        # indexes parameter row zero, so construct that singleton mode as a
        # zero-stride CuTe view instead of materializing a repeated tensor.
        param_layout = cute.make_layout((1, cute.size(mW)), stride=(0, 1))
        mW2 = cute.make_tensor(mW.iterator, param_layout)
        mB2 = cute.make_tensor(mB.iterator, param_layout)
        self.kernel(
            gX,
            tile(mU),
            tile(mW2),
            tile(mB2),
            tile(mY0),
            tile(mY1),
            tile(mY2),
            tile(mM),
            mMean,
            mRstd,
            eps,
            drop,
            thresh,
            thr_layout,
            val_layout,
            cute.size(gX, mode=[1, 1]),
            seed,
            nrows,
            row_off,
            ncols,
        ).launch(
            grid=(grid_n, 1, 1),
            block=(self.rows_per_cta * 32, 1, 1),
            min_blocks_per_mp=self.min_blocks_per_mp,
            stream=stream,
        )


_COMPILED: dict[tuple, object] = {}


def _plan_signature(tensor: torch.Tensor, *, dynamic_rows: bool) -> tuple:
    shape = tuple(tensor.shape)
    if dynamic_rows:
        shape = (None, *shape[1:])
    return shape, tuple(tensor.stride()), tensor.dtype, tensor.device


def _compiled_launch(key: tuple, compile_args: tuple):
    """Compile once for a plan-time layout contract; runtime N stays symbolic."""
    fn = _COMPILED.get(key)
    if fn is None:
        fn = cute.compile(
            LnMulDropoutForward(),
            *compile_args,
            options="--enable-tvm-ffi",
        )
        _COMPILED[key] = fn
    return fn


def ln_mul_dropout_fwd(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace the Triton mask and ln kernels. Returns (y, mean, rstd, random_mask).

    y is the [N, 3D] concatenation [silu(u) | x | ln*silu(u)], with dropout
    applied independently to each of the three segments. mask is (N, D) int8
    with bit0=y, bit1=x, bit2=u; a set bit means keep. This layout is the
    contract with the Triton backward -- see where the mask is read in
    triton_hstu_linear.py.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be rank 2, got shape {tuple(x.shape)}")
    N, D = x.shape
    if not 1 <= N < (1 << 31):
        raise ValueError(f"N must fit in a positive signed Int32, got {N}")
    if D != 512:
        raise ValueError(f"D must be 512, got {D}")
    if u.shape != x.shape:
        raise ValueError(f"u must have shape {tuple(x.shape)}, got {tuple(u.shape)}")
    if weight.shape != (D,) or bias.shape != (D,):
        raise ValueError(f"weight and bias must have shape ({D},)")
    if x.dtype != torch.bfloat16 or any(tensor.dtype != x.dtype for tensor in (u, weight, bias)):
        raise ValueError("x, u, weight, and bias must all have dtype torch.bfloat16")
    x = x.detach()
    u = u.detach()
    weight = weight.detach()
    bias = bias.detach()
    dev = x.device
    if any(tensor.device != dev for tensor in (u, weight, bias)):
        raise ValueError("x, u, weight, and bias must be on the same device")
    if dev.type != "cuda":
        raise ValueError("x, u, weight, and bias must be CUDA tensors")
    if tuple(x.stride()) != (D, 1):
        raise ValueError(f"x must have stride ({D}, 1), got {tuple(x.stride())}")
    if u.stride(1) != 1 or u.stride(0) < D or u.stride(0) % 8 != 0:
        raise ValueError("u must have unit innermost stride and 16-byte-aligned, non-overlapping rows")
    if tuple(weight.stride()) != (1,) or tuple(bias.stride()) != (1,):
        raise ValueError("weight and bias must be contiguous")
    if any(tensor.data_ptr() % ALIGN != 0 for tensor in (x, u, weight, bias)):
        raise ValueError(f"all inputs must be {ALIGN}-byte aligned")
    y = torch.empty((N, 3 * D), dtype=x.dtype, device=dev)
    mask = torch.empty((N, D), dtype=torch.int8, device=dev)
    mean = torch.empty((N,), dtype=torch.float32, device=dev)
    rstd = torch.empty((N,), dtype=torch.float32, device=dev)

    if training:
        effective_dropout_ratio = dropout_ratio
        thresh = _keep_threshold32(dropout_ratio) if PHILOX32 else _keep_threshold(dropout_ratio)
    else:
        effective_dropout_ratio = 0.0
        thresh = 0
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)

    launch_args = (
        x,
        u,
        weight,
        bias,
        y[:, :D],
        y[:, D : 2 * D],
        y[:, 2 * D :],
        mask,
        mean,
        rstd,
        cutlass.Int64(seed),
        cutlass.Int32(N),
        cutlass.Int32(0),
        cutlass.Float32(eps),
        cutlass.Float32(effective_dropout_ratio),
        cutlass.Uint32(thresh),
        cutlass.Int32(D),
        stream,
    )
    rows = cute.sym_int()
    dtype = _convert_to_cutlass_data_type(x.dtype)
    compile_args = (
        make_fake_tensor(dtype, (rows, D), stride=tuple(x.stride()), assumed_align=ALIGN),
        make_fake_tensor(dtype, (rows, D), stride=tuple(u.stride()), assumed_align=ALIGN),
        make_fake_tensor(dtype, (D,), stride=tuple(weight.stride()), assumed_align=ALIGN),
        make_fake_tensor(dtype, (D,), stride=tuple(bias.stride()), assumed_align=ALIGN),
        make_fake_tensor(dtype, (rows, D), stride=(3 * D, 1), assumed_align=ALIGN),
        make_fake_tensor(dtype, (rows, D), stride=(3 * D, 1), assumed_align=ALIGN),
        make_fake_tensor(dtype, (rows, D), stride=(3 * D, 1), assumed_align=ALIGN),
        make_fake_tensor(cutlass.Int8, (rows, D), stride=(D, 1), assumed_align=ALIGN),
        make_fake_tensor(cutlass.Float32, (rows,), stride=(1,), assumed_align=4),
        make_fake_tensor(cutlass.Float32, (rows,), stride=(1,), assumed_align=4),
        cutlass.Int64(0),
        cutlass.Int32(1),
        cutlass.Int32(0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Uint32(0),
        cutlass.Int32(D),
        make_fake_stream(use_tvm_ffi_env_stream=False),
    )
    key = (
        "hstu_lmsd_fwd_dynamic_v1",
        _plan_signature(x, dynamic_rows=True),
        _plan_signature(u, dynamic_rows=True),
        _plan_signature(weight, dynamic_rows=False),
        _plan_signature(bias, dynamic_rows=False),
        HOIST,
        DIV_MODE,
        TPR,
        VEC,
        ROWS,
        TV_ROW,
        PACK_TAIL,
        REREAD_X,
        PHILOX32,
        MIN_BLOCKS_PER_MP,
    )
    with torch.cuda.device(dev):
        compiled = _compiled_launch(key, compile_args)
    compiled(*launch_args)
    return y, mean, rstd, mask


# Measured and rejected, at the production shape on Rubin sm_107a. Kept as a
# list so these are not re-derived; the numbers are in the MR discussion.
#
#   TMA into shared memory, seven pipeline shapes    1.727 - 2.36 ms
#     dedicated DMA warp, dedicated epilogue warp, both, unspecialised
#     software pipelining, per-warp pipelines, multi-row TMA tiles. All lose:
#     a warp spent moving data is a warp not issuing arithmetic, and every KB
#     of staging costs a resident CTA, in a kernel bound by instruction issue.
#   Storing y through shared memory + one TMA        2.007 ms
#     the six stores do not disappear, they change destination.
#   Several warps per row (TPR < 32 the other way)   2.03 - 2.98 ms
#     the reduction leaves the warp and the partials need smem + a barrier.
#   Capping registers to raise occupancy             no effect
#     118 registers regardless of min_blocks_per_mp, PtxasOptions, or
#     CUTE_DSL_COMPILER_OPT=ptx-options=-maxrregcount. The option reaches the
#     compile pipeline; the backend does not honour it.
#   cute.arch.mul_wide for the Philox products       neutral
#     ptxas already fuses mul_hi + mul into the same instruction count.
#   cute.arch.bfe / lop3 for the mask path           neutral / -1.6%
#     same story: ptxas already emits BFE and LOP3 there, and pinning them by
#     hand only removes scheduling freedom. Together with mul_wide this closes
#     the "hand-write it with Rubin instructions" direction on the integer
#     side; the float side (bf16x2 FMA, ex2.bf16x2) is closed by precision.
#   Deferring u's load, or the output partitions     -28%, -0.7%
#     deferring a load exposes its latency; deferring address arithmetic just
#     recomputes it. Neither frees registers: 118 is the live data itself.
#
# Not pursued because it changes the algorithm: packing the three Bernoulli
# planes into subfields of one 32-bit word cuts Philox calls 33% and measured
# 1.539 ms, at the cost of the keep probability moving from 0.900030 to
# 0.900391 and the mask no longer matching the reference bit for bit.
