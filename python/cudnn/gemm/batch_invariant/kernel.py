# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted for the cuDNN Frontend compiler and launcher from FlashInfer's
# blocks/ and fixup.py in flashinfer/gemm/batch_invariant/ (e2d163fb3d46).

"""BF16 forward GEMM with fixed K16 arithmetic and an optional column bias."""

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64

from cudnn.frost.compiled_cache import compile_cached
from cudnn.frost.device import device_context

from ._fixup import reduce_partials
from ._quantized.kernel import BiasedBlockGemm, BlockGemm
from ._schedule import select_schedule


class _Gemm:
    def __init__(self, b_major, chunks, biased, knobs):
        self.b_major = b_major
        self.chunks = chunks
        self.biased = biased
        self.workspace_split = chunks > 1 and not knobs.cta_chunks
        body = BiasedBlockGemm if biased and not self.workspace_split else BlockGemm
        self.body = body(knobs, "bf16", "f32" if self.workspace_split else "bf16", cta_chunks=knobs.cta_chunks)

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
        resident: Int32,
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
            bias = cute.make_tensor(cute.make_ptr(BFloat16, bias_addr, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((n,)))
        if cutlass.const_expr(self.workspace_split):
            partials = cute.make_tensor(
                cute.make_ptr(Float32, workspace_addr, cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout((m, n, chunks), stride=(n_stride, 1, Int64(m) * n_stride)),
            )
            self.body(a, b, partials, resident, stream)
            reduce_partials(chunks, partials, out, None, None, bias, stream)
        else:
            self.body._launch(a, b, out, resident, stream, None, None, bias)


@dataclass(frozen=True)
class _Compiled:
    launch: object
    args: tuple
    workspace_bytes: int

    def execute(self, a, b, out, bias, workspace, stream):
        self.launch(a, b, out, bias or 0, workspace or 0, *self.args, cuda.CUstream(stream))


def build(m, n, k, lda, b_major, bias, device):
    cfg = select_schedule(m, n, k, b_major, device)
    knobs, chunks = cfg.knobs, cfg.chunks
    with device_context(device):
        launch = compile_cached(
            _Gemm(b_major, chunks, bias, knobs),
            *([Int64(0)] * 5),
            Int32(0),
            Int32(0),
            Int32(0),
            Int64(0),
            Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            cache_key=f"batch_invariant_persistent_bf16_v2_{b_major}_{chunks}_{int(bias)}_{knobs}",
            symbol="batch_invariant_gemm_bf16",
            options="--enable-tvm-ffi",
        )
    workspace_bytes = chunks * m * n * 4 if chunks > 1 and not knobs.cta_chunks else 0
    return _Compiled(launch, (m, n, k, lda, cfg.resident_clusters), workspace_bytes)
