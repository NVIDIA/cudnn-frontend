# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch allocating wrappers for prepared blk64 pairs and split 64-row TMA.

The normal blk64 kernels do not import this module. All main MMA tiles stay
128x128x128. Backward retains KV ownership and loops over original Q64 ids.
"""

import torch
import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda

from cudnn._torch_stream import record_streams
from cudnn.frost.compiled_cache import compile_cached
from ._blk64_reorder import reorder
from ._interface import _to_cute_tensor_with_dynamic_modes
from .csrc.utils.cute_dsl_utils import to_cute_tensor
from .csrc.fwd.sm100_blk128.bsa_fwd_sm100 import BlockSparseAttnForwardSm100Blk128
from .csrc.bwd.sm100_blk128.bsa_bwd_sm100 import BlockSparseAttnBackwardSm100Blk128, BsaK2qCsrTensors
from .csrc.bwd.bsa_bwd_prepost import _bwd_preprocess, _bwd_postprocess_convert


@cute.kernel
def _post_dq_kernel(accum, output, scale: cutlass.Float32):
    block, head, batch = cute.arch.block_idx()
    thread = cute.arch.thread_idx()[0]
    for i in cutlass.range_constexpr(32):
        flat = thread + i * 256
        row, column = flat // 128, flat % 128
        # Two independent Q64 reductions from the unchanged M128 TMEM tile.
        src = cutlass.Int64(block) * 8192 + (column // 32) * 2048 + (column % 32 // 4) * 256 + row * 4 + column % 4
        output[batch, block * 64 + row, head, column] = (accum[batch, head, src] * scale).to(output.element_type)


_post_dq_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _post_dq(accum, output, scale: cutlass.Float32, stream):
    _post_dq_kernel(accum, output, scale).launch(grid=(output.shape[1] // 64, output.shape[2], output.shape[0]), block=256, stream=stream)


@cute.kernel
def _init_tail_kernel(lse, delta):
    head, batch, _ = cute.arch.block_idx()
    row = lse.shape[2] - 64 + cute.arch.thread_idx()[0]
    lse[batch, head, row] = cutlass.Float32(float("inf"))
    delta[batch, head, row] = cutlass.Float32(0.0)


_init_tail_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _init_tail(lse, delta, stream):
    _init_tail_kernel(lse, delta).launch(grid=(lse.shape[1], lse.shape[0], 1), block=64, stream=stream)


def _signature(args):
    # Shapes/strides/dtypes and optional declarations only; no data or pointers.
    return tuple(
        (tuple(x.shape), tuple(x.stride()), str(x.dtype)) if isinstance(x, torch.Tensor) else type(x).__name__ if x is not None else None for x in args
    )


def _simple_compile(kernel, args, name):
    return compile_cached(
        kernel,
        *[to_cute_tensor(x) if isinstance(x, torch.Tensor) else x for x in args],
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        cache_key=f"bsa_blk64_paired:{name}:{_signature(args)}",
        options="--enable-tvm-ffi",
    )


def _permute(tensor, permutation, identity, scatter=False):
    if identity:
        return tensor
    result = torch.empty_like(tensor)
    reorder(tensor, result, permutation, scatter=scatter)
    return result


def _record(plan, *tensors):
    record_streams(
        (*tensors, plan._q_permutation, plan._kv_permutation, plan._routes, plan._counts, plan._offsets, plan._edges),
        None,
        plan._source.device,
    )


def forward(q, k, v, plan, scale):
    _record(plan, q, k, v)
    scale = 128**-0.5 if scale is None else float(scale)
    qp = _permute(q, plan._q_permutation, plan._q_identity)
    op = torch.empty_like(q)
    lp = torch.empty((q.shape[0], q.shape[2], q.shape[1]), device=q.device, dtype=torch.float32)
    args = (qp.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), op.transpose(1, 2), lp, scale, plan._routes, None, plan._pair_count, plan._counts)
    fn = compile_cached(
        BlockSparseAttnForwardSm100Blk128(128, has_block_sizes=False, split_kv_load64=True),
        *[
            (
                _to_cute_tensor_with_dynamic_modes(x, dynamic_modes=tuple(range(x.ndim)), assumed_align=16 if x.ndim == 4 else 4)
                if isinstance(x, torch.Tensor)
                else x
            )
            for x in args
        ],
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        cache_key=f"bsa_blk64_paired:fwd:{_signature(args)}",
        options="--enable-tvm-ffi",
    )
    fn(*args, cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream))
    return _permute(op, plan._q_permutation, plan._q_identity, scatter=True), _permute(lp, plan._q_permutation, plan._q_identity, scatter=True)


def backward(dout, q, k, v, out, lse, plan, scale, dq, dk, dv):
    _record(plan, dout, q, k, v, out, lse, dq, dk, dv)
    scale = 128**-0.5 if scale is None else float(scale)
    b, n, h, _ = q.shape
    kp, vp = [_permute(t, plan._kv_permutation, plan._kv_identity) for t in (k, v)]
    dq = torch.empty_like(q) if dq is None else dq
    dk = torch.empty_like(k) if dk is None else dk
    dv = torch.empty_like(v) if dv is None else dv
    dkp, dvp = (dk, dv) if plan._kv_identity else (torch.empty_like(k), torch.empty_like(v))
    da = torch.empty((b, h, n * 128), device=q.device, dtype=torch.float32)
    l2, delta = [torch.empty((b, h, n + 64), device=q.device, dtype=torch.float32) for _ in range(2)]
    multiple = plan._offsets.shape[2] > 1
    dka, dva = [torch.empty((b, h, k.shape[1] * 128), device=q.device, dtype=torch.float32) for _ in range(2)] if multiple else (dkp, dvp)
    args = (q, kp, vp, dout, l2, delta, da, dka, dva, scale)
    csr = (plan._offsets, plan._edges)
    fn = compile_cached(
        BlockSparseAttnBackwardSm100Blk128(128, multiple, split_q_load64=True),
        *[to_cute_tensor(x) if isinstance(x, torch.Tensor) else x for x in args],
        BsaK2qCsrTensors(*[to_cute_tensor(x, assumed_align=4) for x in csr]),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        cache_key=f"bsa_blk64_paired:bwd:{_signature((*args, *csr))}",
        options="--enable-tvm-ffi",
    )
    tail = _simple_compile(_init_tail, (l2, delta), "tail")
    post = _simple_compile(_post_dq, (da, dq, scale), "dq")
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    # Reset on EVERY execution/replay. The sentinel makes an odd last Q64
    # neighbor contribute zero; no global Q/dO/O/LSE permutation is required.
    tail(l2, delta, stream)
    _bwd_preprocess(out, dout, delta, lse, l2, da, None, None, None, cutlass.BFloat16, 128, 128, 64)
    if multiple:
        dka.zero_()
        dva.zero_()
    fn(*args, csr, stream)
    post(da, dq, scale, stream)
    if multiple:
        _bwd_postprocess_convert(dka, dkp, scale, None, None, 100, cutlass.BFloat16, 128, 128, False)
        _bwd_postprocess_convert(dva, dvp, 1.0, None, None, 100, cutlass.BFloat16, 128, 128, False)
    if not plan._kv_identity:
        reorder(dkp, dk, plan._kv_permutation, scatter=True)
        reorder(dvp, dv, plan._kv_permutation, scatter=True)
    return dq, dk, dv
