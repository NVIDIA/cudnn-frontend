# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""DSv4.1 mHC projection/RMS input and weight gradients.

Adapted from NVIDIA Megatron-LM's ``_ct_fused_grad_x_weight_kernel`` in
``megatron/core/fusions/fused_mhc_kernels.py`` at commit
25dc53dbb0dbc4d9b884d749e4c432ed88c4b73b (BSD-3-Clause).
The split-M decomposition and fixed-order partial-gradient reduction extend
that implementation. TF32 operand conversion follows the source math policy.
"""

import cuda.tile as ct
import triton
import triton.language as tl


@ct.kernel
def cudnn_mhc_projection_partial(
    X,
    W,
    GP,
    GR,
    R,
    DX,
    PARTIAL,
    M: int,
    N: int,
    K: int,
    TM: ct.Constant[int],
    TN: ct.Constant[int],
    TK: ct.Constant[int],
    SPLITS: ct.Constant[int],
):
    kid = ct.bid(0)
    split = ct.bid(1)
    mtiles = ct.cdiv(M, TM)
    per_split = ct.cdiv(mtiles, SPLITS)
    begin = split * per_split
    end = ct.minimum(begin + per_split, mtiles)
    weight = ct.load(W, index=(0, kid), shape=(TN, TK), padding_mode=ct.PaddingMode.ZERO)
    dw = ct.full((TK, TN), 0.0, dtype=ct.float32)
    n_offsets = ct.arange(TN, dtype=ct.int32)
    valid_n = ct.reshape(n_offsets < N, (1, TN))
    for mid in range(begin, end):
        gp = ct.load(GP, index=(mid, 0), shape=(TM, TN), padding_mode=ct.PaddingMode.ZERO)
        gp = ct.where(valid_n, gp, 0.0)
        x = ct.load(X, index=(mid, kid), shape=(TM, TK), padding_mode=ct.PaddingMode.ZERO)
        gr = ct.load(GR, index=(mid, 0), shape=(TM, 1), padding_mode=ct.PaddingMode.ZERO)
        r = ct.load(R, index=(mid, 0), shape=(TM, 1), padding_mode=ct.PaddingMode.ZERO).astype(ct.float32)
        dx = (gr * (1.0 / (r * K))) * x.astype(ct.float32)
        dx = ct.mma(gp.astype(ct.tfloat32), weight.astype(ct.tfloat32), acc=dx)
        ct.store(DX, index=(mid, kid), tile=dx.astype(DX.dtype))
        dw = ct.mma(x.transpose().astype(ct.tfloat32), gp.astype(ct.tfloat32), acc=dw)
    ct.store(PARTIAL, index=(split, 0, kid), tile=ct.reshape(dw.transpose(), (1, TN, TK)))


@triton.jit
def cudnn_mhc_projection_reduce(PARTIAL, DW, COUNT: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < COUNT
    total = tl.full((BLOCK,), 0.0, tl.float32)
    # Fixed summation order, without atomics. Every partial is produced by the
    # preceding launch on the same CUDA stream, before this kernel can read it.
    for split in range(SPLITS):
        total += tl.load(PARTIAL + split * COUNT + i, valid, 0)
    tl.store(DW + i, total, valid)
