# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allocation/compilation boundary for the native Hopper blk128 kernels."""

import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

_FWD_CACHE = {}
_BWD_CACHE = {}


def _leading_dim(t):
    return next((i for i, stride in enumerate(t.stride()) if stride == 1), None)


def _key(t):
    # Match mark_layout_dynamic: extents and non-unit strides are runtime values.
    return None if t is None else (t.dtype, t.ndim, _leading_dim(t), tuple(s == 0 for s in t.stride()))


def _to_dynamic_tensor(t, assumed_align):
    if t is None:
        return None
    return from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=_leading_dim(t))


def _validate_tma(**tensors):
    for name, t in tensors.items():
        if t.stride(-1) != 1 or t.data_ptr() % 16 or any(s % 8 for n, s in zip(t.shape[:-1], t.stride()[:-1]) if n > 1):
            raise NotImplementedError(f"SM90 blk128 requires 16-byte aligned rows for {name}; got shape={tuple(t.shape)}, strides={t.stride()}")


def _version_gate():
    from ._interface import _cutlass_dsl_version

    if _cutlass_dsl_version() < (4, 6, 2):
        raise RuntimeError(f"SM90 blk128 BSA requires nvidia-cutlass-dsl>=4.6.2; found {cutlass.__version__}")


def forward(q, k, v, indices, count, sizes=None, counts=None, scale=None, layout="bhsd", splits=1):
    _version_gate()
    from .csrc.fwd.sm90_blk128.bsa_fwd_sm90 import BlockSparseAttnForwardSm90Blk128
    from ._interface import _combine_blk64_kv_bucketed_partials

    _validate_tma(q=q, k=k, v=v)
    q, k, v = (q, k, v) if layout == "bhsd" else (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    b, h, sq, d = q.shape
    dv = v.shape[-1]
    if splits == 1:
        out = torch.empty((b, h, sq, dv) if layout == "bhsd" else (b, sq, h, dv), device=q.device, dtype=q.dtype)
        out_bh = out if layout == "bhsd" else out.transpose(1, 2)
    else:
        out_bh = torch.empty((b, h * splits, sq, dv), device=q.device, dtype=torch.float32)
    lse = torch.empty((b, h * splits, sq), device=q.device, dtype=torch.float32)
    args = (
        q.permute(2, 3, 1, 0),
        k.permute(2, 3, 1, 0),
        v.permute(3, 2, 1, 0),
        out_bh.permute(2, 3, 1, 0),
        lse.permute(2, 1, 0),
        indices.permute(3, 2, 1, 0),
        counts.permute(2, 1, 0) if counts is not None else indices,
        sizes if sizes is not None else indices,
    )
    variable = counts is not None
    stages = 1 if not variable and (count + splits - 1) // splits <= 1 else 2
    has_kv_tail = k.shape[2] % 128 != 0
    scale_nonpositive = scale is not None and scale <= 0
    key = (
        q.device.index,
        d,
        dv,
        h // k.shape[1],
        scale_nonpositive,
        variable,
        sizes.ndim if sizes is not None else 0,
        splits,
        stages,
        has_kv_tail,
        *map(_key, args),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    if key not in _FWD_CACHE:
        kernel = BlockSparseAttnForwardSm90Blk128(
            d, dv, h // k.shape[1], variable, sizes.ndim if sizes is not None else 0, splits, stages, scale_nonpositive, has_kv_tail
        )
        _FWD_CACHE[key] = cute.compile(
            kernel,
            *(_to_dynamic_tensor(t, 16 if i < 4 else 4) for i, t in enumerate(args)),
            cutlass.Float32(1.0),
            cutlass.Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )
    with torch.cuda.nvtx.range("cudnn_bsa_sm90_blk128_fwd"):
        _FWD_CACHE[key](*(t.detach() for t in args), d**-0.5 if scale is None else scale, count, stream)
    if splits > 1:
        out, lse = _combine_blk64_kv_bucketed_partials(q, out_bh, lse, splits)
        if layout == "bshd":
            out = out.transpose(1, 2).contiguous()
    return out, lse


def backward(do, q, k, v, o, lse, indices, count, sizes=None, counts=None, scale=None, dq=None, dk=None, dv=None, bucket=None):
    _version_gate()
    from .csrc.bwd.sm90_blk128.bsa_bwd_sm90 import BlockSparseAttnBackwardSm90Blk128
    from ._interface import _empty_bwd_workspace_with_zeroed_accum
    from .csrc.bwd.bucketed_k2q_csr import build_bucketed_k2q_csr_cutedsl

    _validate_tma(q=q, k=k, v=v, o=o, do=do)
    b, h, sq, d = q.shape
    sk = k.shape[2]
    bucket = 256 if bucket is None else bucket
    if bucket <= 0:
        raise ValueError("bucket_size_blocks must be positive")
    offsets, edges, _, _ = build_bucketed_k2q_csr_cutedsl(indices, count, (sk + 127) // 128, bucket_size_blocks=bucket, q2k_block_nums=counts)
    dq = torch.empty_like(q) if dq is None else dq
    dk = torch.empty_like(k) if dk is None else dk
    dv = torch.empty_like(v) if dv is None else dv
    _validate_tma(dq=dq, dk=dk, dv=dv)
    if sizes is not None and sizes.ndim == 1:
        sizes = sizes[None, :].expand(b, -1)
    workspace = _empty_bwd_workspace_with_zeroed_accum(
        batch_size=b,
        num_heads=h,
        seqlen_q=sq,
        seqlen_k=sk,
        head_dim=d,
        round_q_to=128,
        round_k_to=128,
        round_d_to=32,
        zero_dq_accum=False,
        device=q.device,
        zero_kv_accum=offsets.shape[2] != 1,
    )
    args = (do, o, q, k, v, lse, dq, dk, dv, offsets, edges, sizes, workspace)
    scale_nonpositive = scale is not None and scale <= 0
    direct_kv = offsets.shape[2] == 1
    has_kv_tail = sk % 128 != 0
    key = (q.device.index, d, scale_nonpositive, direct_kv, has_kv_tail, *map(_key, args))
    shape = (sq, sk, d, (h, b))
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    if key not in _BWD_CACHE:
        kernel = BlockSparseAttnBackwardSm90Blk128(
            cutlass.BFloat16 if q.dtype == torch.bfloat16 else cutlass.Float16, d, d, scale_nonpositive, direct_kv, has_kv_tail
        )
        cargs = tuple(_to_dynamic_tensor(t, 4 if i in (5, 9, 10, 11) else 16) for i, t in enumerate(args))
        _BWD_CACHE[key] = cute.compile(
            kernel, shape, *cargs, cutlass.Float32(1.0), cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False), options="--enable-tvm-ffi"
        )
    with torch.cuda.nvtx.range("cudnn_bsa_sm90_blk128_bwd"):
        _BWD_CACHE[key](shape, *args, d**-0.5 if scale is None else scale, stream)
    return dq, dk, dv
