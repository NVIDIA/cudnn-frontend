# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Operand views and chunk placements are adapted from FlashInfer's
# flashinfer/gemm/batch_invariant/api.py (e2d163fb3d46) for cuDNN graph execution.

"""Compile fixed-reduction quantized GEMM plans on the graph's CUDA device."""

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64

from cudnn.frost.compiled_cache import compile_cached
from cudnn.frost.device import device_context

from ._fixup import reduce_partials
from ._formats import ALPHA_GIVEN, BLOCK_SCALED, FP8_FMTS, MX_MIXED_FMTS
from ._quantized.blockscale_fp8 import BlockScaleFp8Gemm
from ._quantized.blockscaled import PLANS, BlockScaledGemm, ScaledBlockScaledGemm
from ._quantized.blockscaled_mx_mixed import plan_for as mixed_plan_for
from ._quantized.config import schedule_for
from ._quantized.kernel import BlockGemm, ScaledBlockGemm
from ._quantized.operands import IN_DT, OUT_DT


def _tensor(dtype, address, shape, stride, align=16):
    ptr = cute.make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)
    return cute.make_tensor(ptr, cute.make_layout(shape, stride=stride))


class _Launch:
    def __init__(self, fmt, out_fmt, b_major, chunks, batched, has_alpha, pdl, knobs, native_k):
        self.fmt = fmt
        self.out_fmt = out_fmt
        self.b_major = b_major
        self.chunks = chunks
        self.batched = batched
        self.has_alpha = has_alpha
        self.swap = knobs.swap_ab
        self.input_chunks = 1 if knobs.cluster_k > 1 else chunks
        self.direct = chunks == 1 or knobs.cluster_k > 1 or knobs.cta_chunks
        body_fmt = out_fmt if self.direct else "f32"
        self.a_dtype = self.b_dtype = IN_DT[fmt]
        if fmt in BLOCK_SCALED:
            plan = mixed_plan_for(fmt, knobs) if fmt in MX_MIXED_FMTS else PLANS[fmt]
            a_dt, b_dt = (plan.B_DT, plan.A_DT) if self.swap else (plan.A_DT, plan.B_DT)
            self.packed_a = a_dt.width == 4
            self.packed_b = b_dt.width == 4
            self.a_dtype = cutlass.Uint8 if self.packed_a else a_dt
            self.b_dtype = cutlass.Uint8 if self.packed_b else b_dt
            scaled = fmt in ALPHA_GIVEN and self.direct
            cls = ScaledBlockScaledGemm if scaled else BlockScaledGemm
            self.body = cls(knobs, fmt, body_fmt, cta_chunks=knobs.cta_chunks, plan_cls=plan, enable_pdl=pdl)
        else:
            self.packed_a = self.packed_b = False
            if fmt in FP8_FMTS:
                cls = ScaledBlockGemm if self.direct else BlockGemm
                self.body = cls(knobs, fmt, body_fmt, cta_chunks=knobs.cta_chunks)
            else:
                self.body = BlockScaleFp8Gemm(knobs, body_fmt, native_k=native_k)

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
        release_mask: Int32,
        stream: cuda.CUstream,
    ):
        chunks = self.chunks
        n = cute.assume(n, divby=8)
        # Packed operands enter the body as storage bytes, then the TMA descriptor reinterprets the two E2M1 elements per byte.
        ka = cute.assume(k // (self.input_chunks * (2 if self.packed_a else 1)), divby=16)
        kb = cute.assume(k // (self.input_chunks * (2 if self.packed_b else 1)), divby=16)
        row_a = cute.assume(lda // (2 if self.packed_a else 1), divby=16)
        row_b = cute.assume(ldb // (2 if self.packed_b else 1), divby=16)
        planes = batch if self.batched else self.input_chunks
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
        if cutlass.const_expr(not self.direct):
            c = _tensor(Float32, wp, (m, n, chunks), (cute.assume(Int64(n), divby=8), 1, cute.assume(Int64(m) * Int64(n), divby=8)))
        sa = sb = alpha = None
        if cutlass.const_expr(self.fmt in FP8_FMTS):
            sa = _tensor(Float32, sap, (1,), (1,), 4)
            sb = _tensor(Float32, sbp, (1,), (1,), 4)
            if cutlass.const_expr(self.direct):
                self.body(a, b, c, sa, sb, resident, stream)
            else:
                self.body(a, b, c, resident, stream)
        elif cutlass.const_expr(self.fmt in BLOCK_SCALED):
            # The body derives the full swizzled layouts from operand geometry and the format's block size.
            sa = _tensor(cutlass.Uint8, sap, (1,), (1,))
            sb = _tensor(cutlass.Uint8, sbp, (1,), (1,))
            if cutlass.const_expr(self.has_alpha):
                alpha = _tensor(Float32, alphap, (1,), (1,), 4)
            if cutlass.const_expr(self.swap):
                # Write C^T (or each partial C^T) through a transposed view of the same output storage.
                ct = cute.make_tensor(c.iterator, cute.make_layout((n, m, c.shape[2]), stride=(1, c.stride[0], c.stride[2])))
                self.body._launch(b, a, sb, sa, ct, resident, stream, alpha)
            else:
                self.body._launch(a, b, sa, sb, c, resident, stream, alpha)
        else:
            sa = _tensor(Float32, sap, (m, k // 128), (sa_row, sa_k), 4)
            sb = _tensor(Float32, sbp, ((n - 1) // 128 + 1, k // 128), (sb_row, sb_k), 4)
            self.body(a, b, sa, sb, c, resident, release_mask, stream)
        if cutlass.const_expr(not self.direct):
            if cutlass.const_expr(self.fmt in FP8_FMTS):
                reduce_partials(chunks, c, y, sa, sb, None, stream)
            else:
                reduce_partials(chunks, c, y, alpha, None, None, stream)


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
    grid = schedule_for(op, device)
    fmt, chunks = grid.fmt, grid.chunks
    out_fmt = {"fp16": "f16", "fp32": "f32"}.get(op.out_fmt, op.out_fmt)
    batched = batch > 1 and chunks == 1
    alpha = op.alpha is not None and fmt in ALPHA_GIVEN
    pdl = bool(getattr(op, "enable_pdl", False))
    knobs = grid.knobs
    # Stage consecutive K scales together. Other graph strides use the per-block ring directly.
    native_k = fmt == "e4m3_bs" and op.sa_strides[1] == 1 and op.sb_strides[1] == 1
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
        0,  # Kernel argument for the groupwise scale ring's load-dependent release address.
    )
    key = (fmt, out_fmt, op.b_major, chunks, batched, alpha, pdl, knobs, native_k)
    with device_context(device):
        launch = _Launch(fmt, out_fmt, op.b_major, chunks, batched, alpha, pdl, knobs, native_k)
        kernel = compile_cached(
            launch,
            *([Int64(0)] * 7),
            *([Int32(0)] * 5),
            *([Int64(0)] * 9),
            Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            cache_key=f"batch_invariant_quantized_gemm_v4_{key}",
            symbol="batch_invariant_quantized_gemm",
            options="--enable-tvm-ffi",
        )
    repeat = batch if chunks > 1 else 1
    return _Compiled(
        kernel,
        args,
        repeat,
        (op.a_batch_stride, op.b_batch_stride, op.out_batch_stride * (OUT_DT[out_fmt].width // 8)),
        0 if launch.direct else chunks * m * n * 4,
    )
