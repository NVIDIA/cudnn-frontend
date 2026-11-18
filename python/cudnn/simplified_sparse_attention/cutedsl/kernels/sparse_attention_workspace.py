# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Convert the sparse-backward fp32 dK/dV accumulator to the BF16 gradients."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from . import sparse_attention_backward_tcgen05 as core

BF16 = cutlass.BFloat16
F32 = cutlass.Float32
I32 = cutlass.Int32
I64 = cutlass.Int64
DIM = core.DIM
BR = core.BR
DF = getattr(core, "DF", core.DIM)
NTHREADS = core.NTHREADS
SUPPORTED_HQ = core.SUPPORTED_HQ
SUPPORTED_TOPK = core.SUPPORTED_TOPK
_bwd_kernel = core._bwd_kernel
_compiled = {}
_sm_count = core._sm_count


def _cv(tensor, alignment=16):
    return from_dlpack(tensor.detach(), assumed_align=alignment).mark_layout_dynamic()


@cute.jit
def _converter_tile(w: cute.Tensor, dk: cute.Tensor, dv: cute.Tensor, base: I32, width: cutlass.Constexpr):
    # Every base is a multiple of width. Explicit pointer alignment preserves
    # that fact through the loop's induction variable for vectorized copies.
    wp = cute.make_ptr(
        F32,
        w.iterator.toint() + I64(base) * I64(8),
        cute.AddressSpace.gmem,
        assumed_align=min(32, width * 8),
    )
    kp = cute.make_ptr(
        BF16,
        dk.iterator.toint() + I64(base) * I64(2),
        cute.AddressSpace.gmem,
        assumed_align=min(16, width * 2),
    )
    vp = cute.make_ptr(
        BF16,
        dv.iterator.toint() + I64(base) * I64(2),
        cute.AddressSpace.gmem,
        assumed_align=min(16, width * 2),
    )
    values = cute.make_rmem_tensor(cute.make_layout(width * 2), F32)
    out_k = cute.make_rmem_tensor(cute.make_layout(width), BF16)
    out_v = cute.make_rmem_tensor(cute.make_layout(width), BF16)
    tile = cute.make_tensor(wp, cute.make_layout(width * 2))
    cute.autovec_copy(tile, values)
    for i in cutlass.range_constexpr(width):
        out_k[i] = values[i * 2].to(BF16)
        out_v[i] = values[i * 2 + 1].to(BF16)
    cute.autovec_copy(out_k, cute.make_tensor(kp, cute.make_layout(width)))
    cute.autovec_copy(out_v, cute.make_tensor(vp, cute.make_layout(width)))


@cute.kernel
def _cvt_kernel(
    w: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    n: I32,
    width: cutlass.Constexpr,
    blocks: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    base = (bid * 256 + tid) * width
    if cutlass.const_expr(blocks == 0):
        if base + width <= n:
            _converter_tile(w, dk, dv, base, width)
    else:
        stride = blocks * 256 * width
        while base + width <= n:
            _converter_tile(w, dk, dv, base, width)
            base += stride


@cute.jit
def _launch(
    mQ,
    mK,
    mV,
    mO,
    mDO,
    mLse,
    mIdx,
    mVlen,
    mDQ,
    mW,
    mDK,
    mDV,
    q_token_stride: I32,
    k_token_stride: I32,
    v_token_stride: I32,
    TQ: I32,
    TK: I32,
    nctas: I32,
    hq: cutlass.Constexpr,
    topk: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    kernel_hq = hq
    do_dim = DIM
    cg = tcgen05.CtaGroup.ONE
    KQ_mma = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.K, F32, cg, (BR, kernel_hq))
    dSQ_mma = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.MN, OperandMajorMode.K, F32, cg, (DF, BR))
    dSK_mma = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.MN, OperandMajorMode.MN, F32, cg, (DF, kernel_hq))

    lQs = sm100_utils.make_smem_layout_b(KQ_mma, (BR, kernel_hq, DIM), BF16, 1)
    lDOs = sm100_utils.make_smem_layout_b(KQ_mma, (BR, kernel_hq, do_dim), BF16, 1)
    lKs = sm100_utils.make_smem_layout_a(KQ_mma, (BR, kernel_hq, DIM), BF16, 1)
    lDSs = sm100_utils.make_smem_layout_b(dSQ_mma, (DF, BR, kernel_hq), BF16, 1)
    lQT = sm100_utils.make_smem_layout_a(dSQ_mma, (DF, BR, kernel_hq), BF16, 1)
    lKT = sm100_utils.make_smem_layout_a(dSK_mma, (DF, kernel_hq, BR), BF16, 1)
    lDSmn = sm100_utils.make_smem_layout_b(dSK_mma, (DF, kernel_hq, BR), BF16, 1)

    a128 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, BF16)
    ds_atom_kind = tcgen05.SmemLayoutAtomKind.K_SW64 if kernel_hq == 32 or kernel_hq == 96 else tcgen05.SmemLayoutAtomKind.K_SW32
    a_ds = tcgen05.make_smem_layout_atom(ds_atom_kind, BF16)
    lQ = cute.tile_to_shape(a128, (kernel_hq, DIM), order=(0, 1))
    lDO = cute.tile_to_shape(a128, (kernel_hq, do_dim), order=(0, 1))
    lK = cute.tile_to_shape(a128, (BR, DIM), order=(0, 1))
    lDS = cute.tile_to_shape(a_ds, (BR, kernel_hq), order=(0, 1))

    nelem = TK * DIM

    for group in cutlass.range_constexpr(hq // kernel_hq):
        _bwd_kernel(
            mQ,
            mK,
            mV,
            mO,
            mDO,
            mLse,
            mIdx,
            mVlen,
            mDQ,
            mW,
            KQ_mma,
            dSQ_mma,
            dSK_mma,
            lQ,
            lK,
            lDS,
            lQT,
            lKT,
            lDSmn,
            lQs,
            lKs,
            lDSs,
            lDO,
            lDOs,
            q_token_stride,
            k_token_stride,
            v_token_stride,
            TQ,
            TK,
            nctas,
            kernel_hq,
            hq,
            topk,
            I32(group * kernel_hq),
        ).launch(grid=[nctas, 1, 1], block=[NTHREADS, 1, 1], stream=stream)

    _cvt_kernel(mW, mDK, mDV, nelem, 2, 0).launch(grid=[cute.ceil_div(nelem, 512), 1, 1], block=[256, 1, 1], stream=stream)


@torch.no_grad()
def _run_h96(q, k, v, out, dout, lse, selected_indices, valid_lengths, dq, dk, dv):
    if q.ndim != 3 or q.shape[-1] != DIM:
        raise ValueError(f"q must have shape [Tq, Hq, {DIM}], got {tuple(q.shape)}")
    tq = q.shape[0]
    tk = k.shape[0]
    hq = q.shape[-2]
    topk = selected_indices.shape[-1]
    dev = q.device
    if tq <= 0 or tk <= 0:
        raise ValueError("q and k/v token streams must be nonempty")
    if hq not in SUPPORTED_HQ:
        raise ValueError(f"Hq must be one of {SUPPORTED_HQ}, got {hq}")
    if topk not in SUPPORTED_TOPK:
        raise ValueError(f"topK must be one of {SUPPORTED_TOPK}, got {topk}")
    if k.ndim != 3 or k.shape[1:] != (1, DIM):
        raise ValueError(f"k must have shape [Tk, 1, {DIM}], got {tuple(k.shape)}")
    if v.shape != k.shape:
        raise ValueError(f"v must match k.shape={tuple(k.shape)}, got {tuple(v.shape)}")
    if out.shape != q.shape or dout.shape != q.shape or dq.shape != q.shape:
        raise ValueError("out, dout, and dq must match q.shape")
    if dk.shape != k.shape or dv.shape != k.shape:
        raise ValueError("dk and dv must match k.shape")
    if lse.shape != q.shape[:2] or lse.dtype != torch.float32 or not lse.is_contiguous():
        raise ValueError("lse must be contiguous FP32 [Tq, Hq]")
    if selected_indices.shape != (tq, topk):
        raise ValueError("selected_indices must have shape [Tq, topK]")
    if valid_lengths.shape != (tq,):
        raise ValueError("valid_lengths must have shape [Tq]")
    bf16_tensors = (q, k, v, out, dout, dq, dk, dv)
    if any(tensor.dtype != torch.bfloat16 for tensor in bf16_tensors):
        raise ValueError("q/k/v/out/dout and dq/dk/dv must all be BF16")
    if selected_indices.dtype != torch.int32 or valid_lengths.dtype != torch.int32:
        raise ValueError("selected_indices and valid_lengths must be int32")
    tensors = (*bf16_tensors, lse, selected_indices, valid_lengths)
    if not q.is_cuda or any(tensor.device != dev for tensor in tensors[1:]):
        raise ValueError("all sparse-backward tensors must be on one CUDA device")
    if not all(tensor.is_contiguous() for tensor in (out, dout, selected_indices, valid_lengths, dq, dk, dv)):
        raise ValueError("sparse-backward outputs, state, and support metadata must be contiguous")
    if q.stride()[1:] != (DIM, 1) or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("q/k/v feature rows must be contiguous")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.data_ptr() % 16 != 0 or tensor.stride(0) % 8 != 0:
            raise ValueError(f"{name} base address and token stride must be 16-byte aligned")
    nctas = min(tq, _sm_count(dev))

    stream_id = torch.cuda.current_stream(dev).cuda_stream
    # The interleaved fp32 dK/dV accumulator and the scheduler counter after it live only
    # for this call: every kernel that touches them runs on this stream, so the caching
    # allocator reuses the block only for later work on the same stream.
    w = torch.empty(tk * DIM * 2 + 8, dtype=torch.float32, device=dev)
    w[: tk * DIM * 2].zero_()
    w[tk * DIM * 2 : tk * DIM * 2 + 1].view(torch.int32).fill_(nctas)

    args = (
        # DLPack forbids exporting tensors with ``requires_grad=True``. These
        # detached views share the same storage and preserve the native token
        # strides; gradients are written explicitly to dq/dk/dv below.
        _cv(q),
        _cv(k),
        _cv(v),
        _cv(out.view(-1)),
        _cv(dout.view(-1)),
        _cv(lse.view(-1)),
        _cv(selected_indices.view(-1), 4),
        _cv(valid_lengths.view(-1), 4),
        _cv(dq.view(-1)),
        _cv(w),
        _cv(dk.view(-1)),
        _cv(dv.view(-1)),
        I32(q.stride(0)),
        I32(k.stride(0)),
        I32(v.stride(0)),
    )

    compile_key = (dev.index, hq, topk)
    stream = cuda.CUstream(stream_id)
    if compile_key not in _compiled:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Compile the sparse backward for this head count before graph capture")
        _compiled[compile_key] = cute.compile(
            _launch,
            *args,
            I32(tq),
            I32(tk),
            I32(nctas),
            hq,
            topk,
            make_fake_stream(),
        )
    _compiled[compile_key](*args, I32(tq), I32(tk), I32(nctas), stream)


@torch.no_grad()
def run(q, k, v, out, dout, lse, selected_indices, valid_lengths, dq, dk, dv):
    if q.shape[1] != 96 or BR != 64 or DF != 128:
        return core.run(q, k, v, out, dout, lse, selected_indices, valid_lengths, dq, dk, dv)
    return _run_h96(q, k, v, out, dout, lse, selected_indices, valid_lengths, dq, dk, dv)
