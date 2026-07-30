# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
from functools import lru_cache

import torch

import cutlass.cute as cute
from cutlass import Float32, Int32

from cudnn.block_sparse_attention.csrc.utils.cute_dsl_utils import make_fake_tensor as fake_tensor

from cudnn.block_sparse_attention.csrc.bwd.bsa_bwd_postprocess import BlockSparseAttnBackwardPostprocess
from cudnn.block_sparse_attention.csrc.bwd.bsa_bwd_preprocess import BlockSparseAttnBackwardPreprocess


def _parse_arch_str(arch_str: str) -> int:
    """Parse arch strings like sm_100, SM100, or 100 into integer form."""
    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if not match:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def _get_device_arch_for_device(device_index: int, arch_override: str | None):
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability(device_index)
    return major * 10 + int(minor)


def _get_device_arch():
    return _get_device_arch_for_device(
        torch.cuda.current_device(),
        os.environ.get("CUDNN_BSA_ARCH", None),
    )


def make_fake_bwd_tensors(dtype, has_gqa, varlen_q, varlen_k):
    sym = cute.sym_int
    div = 128 // dtype.width
    b, seqlen_q, seqlen_k, h_q, d, d_v = sym(), sym(), sym(), sym(), sym(), sym()
    h_kv = h_q if not has_gqa else sym()
    seqlen_q_rounded, seqlen_k_rounded = sym(), sym()
    seqlen_q_d_rounded, seqlen_k_d_rounded, seqlen_k_dv_rounded = sym(), sym(), sym()
    total_q, total_k, total_q_rounded, total_k_rounded = sym(), sym(), sym(), sym()
    total_q_d_rounded, total_k_d_rounded, total_k_dv_rounded = sym(), sym(), sym()
    b_seqlenq = (b, seqlen_q) if not varlen_q else (total_q,)
    b_seqlenk = (b, seqlen_k) if not varlen_k else (total_k,)
    mQ = fake_tensor(dtype, (*b_seqlenq, h_q, d), divisibility=div)
    mO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
    mdO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
    mK = fake_tensor(dtype, (*b_seqlenk, h_kv, d), divisibility=div)
    mV = fake_tensor(dtype, (*b_seqlenk, h_kv, d_v), divisibility=div)
    mdQ = fake_tensor(dtype, (*b_seqlenq, h_q, d), divisibility=div)
    mdK = fake_tensor(dtype, (*b_seqlenk, h_kv, d), divisibility=div)
    mdV = fake_tensor(dtype, (*b_seqlenk, h_kv, d_v), divisibility=div)
    if not varlen_q:
        mLSE = fake_tensor(Float32, (b, h_q, seqlen_q), divisibility=1)
        mLSElog2 = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
        mPdPsum = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
        dQaccum = fake_tensor(Float32, (b, h_q, seqlen_q_d_rounded), divisibility=4)
    else:
        mLSE = fake_tensor(Float32, (h_q, total_q), divisibility=1)
        mLSElog2 = fake_tensor(Float32, (h_q, total_q_rounded), divisibility=4)
        mPdPsum = fake_tensor(Float32, (h_q, total_q_rounded), divisibility=4)
        dQaccum = fake_tensor(Float32, (h_q, total_q_d_rounded), divisibility=4)
    if not has_gqa:
        mdKaccum, mdVaccum = None, None
    else:
        if not varlen_k:
            mdKaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_rounded), divisibility=4)
            mdVaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_dv_rounded), divisibility=4)
        else:
            mdKaccum = fake_tensor(Float32, (h_kv, total_k_rounded), divisibility=4)
            mdVaccum = fake_tensor(Float32, (h_kv, total_k_dv_rounded), divisibility=4)
    return mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, dQaccum, mdKaccum, mdVaccum


def _compile_bwd_preprocess(
    dtype,
    head_dim,
    head_dim_v,
    m_block_size,
    has_cuseqlens_q,
    has_seqused_q,
    has_dlse,
    has_dq_accum,
    use_padded_offsets,
):
    """Compile bwd preprocess kernel using fake tensors."""
    mQ, _, _, mO, mdO, _, _, _, mLSE, mLSElog2, mPdPsum, mdQaccum, _, _ = make_fake_bwd_tensors(dtype, has_gqa=True, varlen_q=has_cuseqlens_q, varlen_k=False)
    batch = mQ.shape[0] if not has_cuseqlens_q else cute.sym_int()
    batchp1 = cute.sym_int()
    mCuSeqlensQ = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cuseqlens_q else None
    mSequsedQ = fake_tensor(Int32, (batch,), divisibility=1) if has_seqused_q else None
    mdLSE = fake_tensor(Float32, mLSE.shape, divisibility=1) if has_dlse else None
    mdQaccum = mdQaccum if has_dq_accum else None
    bsa_bwd_pre = BlockSparseAttnBackwardPreprocess(dtype, head_dim, head_dim_v, m_block_size, use_padded_offsets=use_padded_offsets)
    return cute.compile(
        bsa_bwd_pre,
        mO,
        mdO,
        mPdPsum,
        mLSE,
        mLSElog2,
        mdQaccum,
        mCuSeqlensQ,
        mSequsedQ,
        mdLSE,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_preprocess_compile_key(
    arch,
    dtype,
    head_dim,
    head_dim_v,
    m_block_size,
    has_cuseqlens_q,
    has_seqused_q,
    has_dlse,
    has_dq_accum,
    use_padded_offsets,
):
    return (
        int(arch),
        dtype,
        head_dim,
        head_dim_v,
        m_block_size,
        has_cuseqlens_q,
        has_seqused_q,
        has_dlse,
        has_dq_accum,
        use_padded_offsets,
    )


def _bwd_preprocess(
    out,
    dout,
    dpsum,
    lse,
    lse_log2,
    dq_accum,
    cu_seqlens_q,
    seqused_q,
    dlse,
    dtype,
    head_dim,
    head_dim_v,
    m_block_size,
    use_padded_offsets=True,
):
    """Compute dPsum/LSE log2 and optionally zero dQaccum."""
    is_varlen = cu_seqlens_q is not None
    compile_key = _bwd_preprocess_compile_key(
        _get_device_arch(),
        dtype,
        head_dim,
        head_dim_v,
        m_block_size,
        is_varlen,
        seqused_q is not None,
        dlse is not None,
        dq_accum is not None,
        use_padded_offsets,
    )
    if compile_key not in _bwd_preprocess.compile_cache:
        _bwd_preprocess.compile_cache[compile_key] = _compile_bwd_preprocess(*compile_key[1:])
    _bwd_preprocess.compile_cache[compile_key](out, dout, dpsum, lse, lse_log2, dq_accum, cu_seqlens_q, seqused_q, dlse)


_bwd_preprocess.compile_cache = {}


def _compile_bwd_postprocess(
    dtype,
    hdim,
    block_size,
    swap_ab,
    has_cuseqlens_q,
    has_seqused_q,
    arch,
):
    """Compile bwd postprocess kernel using fake tensors."""
    mQ, _, _, _, _, mdQ, _, _, _, _, _, mdQaccum, _, _ = make_fake_bwd_tensors(dtype, has_gqa=True, varlen_q=has_cuseqlens_q, varlen_k=False)
    batch = mQ.shape[0] if not has_cuseqlens_q else cute.sym_int()
    batchp1 = cute.sym_int()
    mCuSeqlensQ = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cuseqlens_q else None
    mSeqUsedQ = fake_tensor(Int32, (batch,), divisibility=1) if has_seqused_q else None
    bsa_bwd_post = BlockSparseAttnBackwardPostprocess(
        dtype,
        hdim,
        arch,
        block_size,
        swap_ab,
    )
    return cute.compile(
        bsa_bwd_post,
        mdQaccum,
        mdQ,
        Float32(0.0),
        mCuSeqlensQ,
        mSeqUsedQ,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_postprocess_convert(
    accum,
    output,
    scale,
    cu_seqlens,
    seqused,
    arch,
    dtype,
    hdim,
    block_size,
    swap_ab,
):
    """Convert float32 backward accumulator to final output dtype."""
    compile_key = (
        dtype,
        hdim,
        block_size,
        swap_ab,
        cu_seqlens is not None,
        seqused is not None,
        arch,
    )
    if compile_key not in _bwd_postprocess_convert.compile_cache:
        _bwd_postprocess_convert.compile_cache[compile_key] = _compile_bwd_postprocess(*compile_key)
    _bwd_postprocess_convert.compile_cache[compile_key](
        accum,
        output,
        scale,
        cu_seqlens,
        seqused,
    )


_bwd_postprocess_convert.compile_cache = {}
