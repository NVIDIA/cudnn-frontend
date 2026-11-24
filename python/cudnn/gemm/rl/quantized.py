# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Compile fixed-reduction quantized GEMM plans on the graph's CUDA device."""

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.cute.math import RoundingMode

from cudnn.frost.compiled_cache import compile_cached
from cudnn.frost.device import compute_capability, device_context, multiprocessor_count

from ._quantized.arithmetic import ALPHA_GIVEN, BLOCK_SCALED, FP8_FMTS, MX_MIXED_FMTS, Knobs
from ._quantized.blockscale_fp8 import BlockScaleFp8Gemm
from ._quantized.blockscaled import PLANS, BlockScaledGemm, ScaledBlockScaledGemm
from ._quantized.blockscaled_mx_mixed import PLANS as MIXED_PLANS
from ._quantized.config import schedule_for
from ._quantized.kernel import BlockGemm, ScaledBlockGemm
from ._quantized.operands import ALIGN, IN_DT, OUT_DT


def _tensor(dtype, address, shape, stride, align=16):
    ptr = cute.make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)
    return cute.make_tensor(ptr, cute.make_layout(shape, stride=stride))


class _Launch:
    def __init__(self, fmt, out_fmt, b_major, chunks, batched, has_alpha, pdl, knobs):
        self.fmt = fmt
        self.out_fmt = out_fmt
        self.b_major = b_major
        self.chunks = chunks
        self.batched = batched
        self.has_alpha = has_alpha
        body_fmt = out_fmt if chunks == 1 else "f32"
        self.a_dtype = self.b_dtype = IN_DT[fmt]
        if fmt in BLOCK_SCALED:
            plan = (MIXED_PLANS if fmt in MX_MIXED_FMTS else PLANS)[fmt]
            self.packed_a = plan.A_DT.width == 4
            self.packed_b = plan.B_DT.width == 4
            self.a_dtype = cutlass.Uint8 if self.packed_a else plan.A_DT
            self.b_dtype = cutlass.Uint8 if self.packed_b else plan.B_DT
            scaled = fmt in ALPHA_GIVEN and chunks == 1
            cls = ScaledBlockScaledGemm if scaled else BlockScaledGemm
            self.body = cls(knobs, fmt, body_fmt, plan_cls=plan, enable_pdl=pdl)
        else:
            self.packed_a = self.packed_b = False
            if fmt in FP8_FMTS:
                cls = ScaledBlockGemm if chunks == 1 else BlockGemm
                self.body = cls(knobs, fmt, body_fmt)
            else:
                self.body = BlockScaleFp8Gemm(knobs, body_fmt)

    @cute.jit
    def __call__(
        self,
        ap: Int64,
        bp: Int64,
        yp: Int64,
        sap: Int64,
        sbp: Int64,
        alphap: Int64,
        wp: Int64,
        m: Int32,
        n: Int32,
        k: Int32,
        batch: Int32,
        resident: Int32,
        lda: Int64,
        ldb: Int64,
        a_batch: Int64,
        b_batch: Int64,
        y_batch: Int64,
        sa_row: Int64,
        sa_k: Int64,
        sb_row: Int64,
        sb_k: Int64,
        stream: cuda.CUstream,
    ):
        chunks = self.chunks
        n = cute.assume(n, divby=8)
        # Packed operands enter the body as storage bytes, then the TMA descriptor reinterprets the two E2M1 elements per byte.
        ka = cute.assume(k // (chunks * (2 if self.packed_a else 1)), divby=16)
        kb = cute.assume(k // (chunks * (2 if self.packed_b else 1)), divby=16)
        row_a = cute.assume(lda // (2 if self.packed_a else 1), divby=16)
        row_b = cute.assume(ldb // (2 if self.packed_b else 1), divby=16)
        planes = batch if self.batched else chunks
        step_a = cute.assume(a_batch, divby=16) if self.batched else cute.assume(Int64(ka), divby=16)
        step_b = cute.assume(b_batch, divby=16) if self.batched else cute.assume(Int64(kb), divby=16)
        b_stride = (row_b, 1, step_b)
        if cutlass.const_expr(self.b_major == "n"):
            b_stride = (1, row_b, step_b if self.batched else Int64(kb) * row_b)
        a = _tensor(self.a_dtype, ap, (m, ka, planes), (row_a, 1, step_a))
        b = _tensor(self.b_dtype, bp, (n, kb, planes), b_stride)
        y_stride = cute.assume(y_batch, divby=8) if self.batched else cute.assume(Int64(n), divby=8)
        y = _tensor(OUT_DT[self.out_fmt], yp, (m, n, batch if self.batched else 1), (cute.assume(Int64(n), divby=8), 1, y_stride))
        c = y
        if cutlass.const_expr(chunks > 1):
            c = _tensor(Float32, wp, (m, n, chunks), (cute.assume(Int64(n), divby=8), 1, cute.assume(Int64(m) * Int64(n), divby=8)))
        sa = sb = alpha = None
        if cutlass.const_expr(self.fmt in FP8_FMTS):
            sa = _tensor(Float32, sap, (1,), (1,), 4)
            sb = _tensor(Float32, sbp, (1,), (1,), 4)
            if cutlass.const_expr(chunks == 1):
                self.body(a, b, c, sa, sb, resident, stream)
            else:
                self.body(a, b, c, resident, stream)
        elif cutlass.const_expr(self.fmt in BLOCK_SCALED):
            # The body derives the full swizzled layouts from operand geometry and the format's block size.
            sa = _tensor(cutlass.Uint8, sap, (1,), (1,))
            sb = _tensor(cutlass.Uint8, sbp, (1,), (1,))
            if cutlass.const_expr(self.has_alpha):
                alpha = _tensor(Float32, alphap, (1,), (1,), 4)
            if cutlass.const_expr(self.fmt in ALPHA_GIVEN and chunks == 1):
                self.body._launch(a, b, sa, sb, c, resident, stream, alpha)
            else:
                self.body(a, b, sa, sb, c, resident, stream)
        else:
            sa = _tensor(Float32, sap, (m, k // 128), (sa_row, sa_k), 4)
            sb = _tensor(Float32, sbp, ((n - 1) // 128 + 1, k // 128), (sb_row, sb_k), 4)
            self.body(a, b, sa, sb, c, resident, stream)
        if cutlass.const_expr(chunks > 1):
            rows = cutlass.min(m, Int32(65535))
            self._reduce(c, y, sa, sb, alpha, rows).launch(grid=((n - 1) // 256 + 1, rows, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def _reduce(self, partials, y, sa, sb, alpha, rows: Int32):
        bx, by, _ = cute.arch.block_idx()
        col = bx * 256 + cute.arch.thread_idx()[0]
        scale = Float32(1)
        if cutlass.const_expr(self.fmt in FP8_FMTS):
            scale = cute.math.mul(sa[0], sb[0], rounding=RoundingMode.NEAREST_EVEN)
        elif cutlass.const_expr(self.has_alpha):
            scale = alpha[0]
        if col < y.shape[1]:
            for i in cutlass.range((y.shape[0] - 1 - by) // rows + 1, unroll=1):
                row = by + i * rows
                total = Float32(0)
                for chunk in cutlass.range_constexpr(self.chunks):
                    total = cute.math.add(total, partials[row, col, chunk], rounding=RoundingMode.NEAREST_EVEN)
                if cutlass.const_expr(self.fmt in FP8_FMTS or self.fmt in ALPHA_GIVEN):
                    total = cute.math.mul(total, scale, rounding=RoundingMode.NEAREST_EVEN)
                y[row, col, 0] = total.to(y.element_type)


@dataclass(frozen=True)
class _Compiled:
    kernel: object
    args: tuple
    repeat: int
    byte_steps: tuple
    workspace_bytes: int

    def execute(self, a, b, out, sa, sb, alpha, workspace, stream):
        stream = cuda.CUstream(stream)
        for i in range(self.repeat):
            self.kernel(
                a + i * self.byte_steps[0], b + i * self.byte_steps[1], out + i * self.byte_steps[2], sa, sb, alpha or 0, workspace or 0, *self.args, stream
            )


def build(op, device):
    m, n, k, batch = op.m, op.n, op.k, op.batch
    grid = schedule_for(op, compute_capability(device), multiprocessor_count(device))
    fmt, chunks = grid.fmt, grid.chunks
    out_fmt = {"fp16": "f16", "fp32": "f32"}.get(op.out_fmt, op.out_fmt)
    batched = batch > 1 and chunks == 1
    alpha = op.alpha is not None and fmt in ALPHA_GIVEN
    pdl = bool(getattr(op, "enable_pdl", False))
    knobs = Knobs(grid.tile, grid.cluster, grid.cluster[0] == 2)
    resident = grid.resident_clusters
    args = (
        m,
        n,
        k,
        batch if batched else 1,
        resident,
        op.lda,
        op.ldb,
        op.a_batch_stride,
        op.b_batch_stride,
        op.out_batch_stride,
        *(op.sa_strides if fmt == "e4m3_bs" else (0, 0)),
        *(op.sb_strides if fmt == "e4m3_bs" else (0, 0)),
    )
    key = (fmt, out_fmt, op.b_major, chunks, batched, alpha, pdl, knobs.mma_tiler_mn, knobs.cluster_shape_mn)
    with device_context(device):
        kernel = compile_cached(
            _Launch(fmt, out_fmt, op.b_major, chunks, batched, alpha, pdl, knobs),
            *([Int64(0)] * 7),
            *([Int32(0)] * 5),
            *([Int64(0)] * 9),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            cache_key=f"rl_quantized_gemm_{key}",
            symbol="rl_quantized_gemm",
            options="--enable-tvm-ffi",
        )
    repeat = batch if chunks > 1 else 1
    return _Compiled(
        kernel,
        args,
        repeat,
        (op.a_batch_stride, op.b_batch_stride, op.out_batch_stride * (OUT_DT[out_fmt].width // 8)),
        chunks * m * n * 4 if chunks > 1 else 0,
    )
