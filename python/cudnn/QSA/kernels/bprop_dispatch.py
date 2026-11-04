# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep nonfinite values in one sparse query out of the queries batched with it."""

import hashlib
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32
from cutlass.cute.runtime import make_fake_stream
from cuda.bindings import driver as cuda

from cudnn._cutlass_compat import SmemAllocator
from cudnn.frost.compiled_cache import compile_cached, template_key
from . import bprop_helpers, bprop_sm100, bprop_sm100_base, fprop_prepare

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SOURCE_DIGESTS = tuple(m.FROST_SOURCE_DIGEST for m in (bprop_helpers, bprop_sm100, bprop_sm100_base, fprop_prepare))


@cute.kernel
def frost_qsa_nonfinite_reduce(partials, nonfinite):
    tid = Int32(cute.arch.thread_idx()[0])
    bad = Int32(0)
    for index in cutlass.range(Int64(tid), partials.shape[0], 256):
        bad = bad | partials[index]
    bad = cute.arch.warp_reduction_max(bad)
    scratch = SmemAllocator().allocate_tensor(Int32, cute.make_layout(8), byte_alignment=16)
    if tid % 32 == 0:
        scratch[tid // 32] = bad
    cute.arch.sync_threads()
    if tid == 0:
        for warp in cutlass.range_constexpr(8):
            bad = bad | scratch[warp]
        nonfinite[0] = bad


frost_qsa_nonfinite_reduce.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _dispatch(
    q, k, v, grad, out, lse, delta, aq, ak, av, ids, counts, lo, hi, first, last, dq, dk, dv, flags, scale: Float32, sms: Int32, stream: cuda.CUstream
):
    flat_k = cute.make_tensor(k.iterator, cute.make_layout((cute.size(k),)))
    flat_v = cute.make_tensor(v.iterator, cute.make_layout((cute.size(v),)))
    pieces = cute.ceil_div(Int64(cute.size(k)), 1024) * 4
    partials = cute.make_tensor(flags.iterator, cute.make_layout((pieces,)))
    nonfinite = cute.make_tensor(flags.iterator + pieces, cute.make_layout(1))
    fprop_prepare.frost_qsa_scan_kv(flat_k, flat_v, partials).launch(grid=(cute.ceil_div(cute.size(k), 1024), 1, 1), block=(128, 1, 1), stream=stream)
    frost_qsa_nonfinite_reduce(partials, nonfinite).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)
    blocks = cute.math.max(Int32(1), cute.math.min(sms * 8, Int32(cute.ceil_div(cute.size(q) // 8, 256))))
    bprop_helpers.prepare_host(out, grad, delta, aq, ak, av, blocks, stream)
    bprop_sm100._bprop_host(
        q, k, v, grad, lse, delta, aq, ak, av, ids, counts, lo, hi, first, last, scale, cute.math.min(Int32(counts.shape[0]), sms), stream, nonfinite
    )
    bprop_sm100_base._bprop_host(
        q, k, v, grad, lse, delta, aq, ak, av, ids, counts, lo, hi, first, last, scale, cute.math.min(Int32(q.shape[0]), sms), stream, nonfinite
    )
    bprop_helpers.finish_host(aq, ak, av, dq, dk, dv, blocks, stream)


def compile_kernel(arch, hq, hkv):
    """Compile the two-launch backward for one batched head geometry (12/1 or 6/1); tensor extents stay dynamic."""
    fake = bprop_helpers.fake
    total, groups, capacity, blocks, flag_size = (cute.sym_int64() for _ in range(5))
    q, kv = fake(cutlass.BFloat16, (total, hq, 256)), fake(cutlass.BFloat16, (total, hkv, 256))
    aq, ak = fake(Float32, (total, hq, 256)), fake(Float32, (total, hkv, 256))
    rows, spans = fake(Int32, (total,), 4), fake(Int32, (blocks,), 4)
    args = (
        q,
        kv,
        kv,
        q,
        aq,
        fake(Float32, (hq, total), 4),
        fake(Float32, (total, hq)),
        aq,
        ak,
        ak,
        fake(Int32, (groups, capacity), 4),
        fake(Int32, (groups,), 4),
        rows,
        rows,
        spans,
        spans,
        q,
        kv,
        kv,
        fake(Int32, (flag_size,), 4),
        Float32(1),
        Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=False),
    )
    return compile_cached(
        _dispatch,
        *args,
        cache_key=template_key(globals(), {"arch": arch, "hq": hq, "hkv": hkv, "sources": SOURCE_DIGESTS}, "compile_kernel"),
        symbol="qsa_backward_dispatch",
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch}a",
    )


def launch(compiled, grad, q, k, v, out, lse, ids, counts, lo, hi, first, last, dq, dk, dv, scratch, scale, stream, sms):
    flags = torch.empty(((k.numel() + 1023) // 1024) * 4 + 1, dtype=torch.int32, device=q.device)
    compiled(
        q,
        k,
        v,
        grad,
        out,
        lse,
        scratch["delta"],
        scratch["dq_acc"],
        scratch["dk_acc"],
        scratch["dv_acc"],
        ids,
        counts,
        lo,
        hi,
        first,
        last,
        dq,
        dk,
        dv,
        flags,
        Float32(scale),
        Int32(sms),
        stream,
    )
