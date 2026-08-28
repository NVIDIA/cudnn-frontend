# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""One-launch BF16/FP32 to MXFP8 staging for fixed training resources."""

from __future__ import annotations

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Float32, Int32

from ..cutedsl_src.helpers.constants import Fp32Max, Fp8E4M3RcpLimit
from ..cutedsl_src.helpers.ptx_helpers import cvt_f32_to_fp8_to_f32


class Mxfp8TrainingStageKernel:
    """Quantize one token row per CTA and repack its routing metadata."""

    _threads_per_cta = 128
    _sf_vec = 32

    def __init__(self, hidden: int, top_k: int) -> None:
        self.hidden = int(hidden)
        self.top_k = int(top_k)
        if self.hidden <= 0 or self.hidden % self._sf_vec:
            raise ValueError(
                "MXFP8 training stage requires hidden divisible by 32"
            )
        if self.top_k <= 0 or self.top_k > self._threads_per_cta:
            raise ValueError(
                "MXFP8 training stage requires "
                f"1 <= top_k <= {self._threads_per_cta}"
            )

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        output: cute.Tensor,
        output_sf: cute.Tensor,
        output_topk_idx: cute.Tensor,
        output_topk_weights: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        self._kernel(
            source,
            topk_idx,
            topk_weights,
            output,
            output_sf,
            output_topk_idx,
            output_topk_weights,
        ).launch(
            grid=[source.shape[0], 1, 1],
            block=[self._threads_per_cta, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def _kernel(
        self,
        source: cute.Tensor,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        output: cute.Tensor,
        output_sf: cute.Tensor,
        output_topk_idx: cute.Tensor,
        output_topk_weights: cute.Tensor,
    ) -> None:
        token = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]
        hidden: cutlass.Constexpr[int] = self.hidden
        sf_vec: cutlass.Constexpr[int] = self._sf_vec
        threads: cutlass.Constexpr[int] = self._threads_per_cta
        block_count: cutlass.Constexpr[int] = hidden // sf_vec
        rounds: cutlass.Constexpr[int] = (
            block_count + threads - 1
        ) // threads

        for block_round in cutlass.range_constexpr(rounds):
            block = tid + Int32(block_round * threads)
            if block < Int32(block_count):
                values = cute.make_rmem_tensor((sf_vec,), Float32)
                absmax = Float32(0.0)
                for element in cutlass.range_constexpr(sf_vec):
                    value = Float32(
                        source[
                            token,
                            block * Int32(sf_vec) + Int32(element),
                        ]
                    )
                    values[element] = value
                    absmax = cute.arch.fmax(
                        absmax,
                        cute.arch.fmax(value, -value),
                    )

                scale_f32 = Float32(
                    cvt_f32_to_fp8_to_f32(
                        absmax * Float32(Fp8E4M3RcpLimit),
                        cutlass.Float8E8M0FNU,
                    )
                )
                scale = scale_f32.to(cutlass.Float8E8M0FNU)
                reciprocal = cute.arch.fmin(
                    cute.arch.rcp_approx(scale_f32),
                    Float32(Fp32Max),
                )
                reciprocal = reciprocal * cute.arch.fmin(
                    scale_f32 * Float32(1.0e30),
                    Float32(1.0),
                )
                for element in cutlass.range_constexpr(sf_vec):
                    output[
                        token,
                        block * Int32(sf_vec) + Int32(element),
                    ] = (
                        values[element] * reciprocal
                    ).to(cutlass.Float8E4M3FN)
                output_sf[token, block] = scale

        if tid < Int32(self.top_k):
            output_topk_idx[token, tid] = Int32(topk_idx[token, tid])
            output_topk_weights[token, tid] = Float32(
                topk_weights[token, tid]
            )


__all__ = ["Mxfp8TrainingStageKernel"]
