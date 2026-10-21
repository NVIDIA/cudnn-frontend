# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QSA backward reduction and output conversion."""

import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import primitives as prims
from cuda.bindings import driver as cuda

from cudnn.frost.compiled_cache import compile_cached, template_key

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

I32, F32, BF16 = cutlass.Int32, cutlass.Float32, cutlass.BFloat16
THREADS, D = 256, 256


@cute.kernel
def frost_qsa_bprop_prepare(out: cute.Tensor, dout: cute.Tensor, delta: cute.Tensor, dq: cute.Tensor, dk: cute.Tensor, dv: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    blocks = cute.arch.grid_dim()[0]
    lane, warp = tid % 32, tid // 32
    po, pdo, pd = out.iterator.raw_ptr(), dout.iterator.raw_ptr(), delta.iterator.raw_ptr()
    rows = I32(out.shape[0] * out.shape[1])
    for row in cutlass.range(bid * 8 + warp, rows, blocks * 8, unroll=1):
        o = (po + row * D + lane * 8).load(count=8, alignment=16)
        do = (pdo + row * D + lane * 8).load(count=8, alignment=16).to(F32)
        value = F32(0)
        for j in cutlass.range_constexpr(8):
            value = value + o[j] * do[j]
        for j in cutlass.range_constexpr(5):
            value = value + prims.shfl_sync(0xFFFFFFFF, value, 16 >> j, 0x1F, prims.Shfl.BFLY)
        if lane == 0:
            (pd + row).store(value)
    zero = cutlass.Vector.from_elements([F32(0)] * 8, F32)
    pq, pk, pv = dq.iterator.raw_ptr(), dk.iterator.raw_ptr(), dv.iterator.raw_ptr()
    for item in cutlass.range(bid * THREADS + tid, rows * D // 8, blocks * THREADS, unroll=1):
        (pq + item * 8).store(zero, alignment=16)
    kv_items = I32(dk.shape[0] * dk.shape[1] * D // 8)
    for item in cutlass.range(bid * THREADS + tid, kv_items, blocks * THREADS, unroll=1):
        (pk + item * 8).store(zero, alignment=16)
        (pv + item * 8).store(zero, alignment=16)


frost_qsa_bprop_prepare.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def prepare_host(
    out: cute.Tensor, dout: cute.Tensor, delta: cute.Tensor, dq: cute.Tensor, dk: cute.Tensor, dv: cute.Tensor, blocks: I32, stream: cuda.CUstream
):
    frost_qsa_bprop_prepare(out, dout, delta, dq, dk, dv).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)


@cute.kernel
def frost_qsa_bprop_finish(aq: cute.Tensor, ak: cute.Tensor, av: cute.Tensor, dq: cute.Tensor, dk: cute.Tensor, dv: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    blocks = cute.arch.grid_dim()[0]
    paq, pak, pav = aq.iterator.raw_ptr(), ak.iterator.raw_ptr(), av.iterator.raw_ptr()
    pq, pk, pv = dq.iterator.raw_ptr(), dk.iterator.raw_ptr(), dv.iterator.raw_ptr()
    q_items = I32(dq.shape[0] * dq.shape[1] * D // 8)
    for item in cutlass.range(bid * THREADS + tid, q_items, blocks * THREADS, unroll=1):
        (pq + item * 8).store((paq + item * 8).load(count=8, alignment=16).to(BF16), alignment=16)
    kv_items = I32(dk.shape[0] * dk.shape[1] * D // 8)
    for item in cutlass.range(bid * THREADS + tid, kv_items, blocks * THREADS, unroll=1):
        (pk + item * 8).store((pak + item * 8).load(count=8, alignment=16).to(BF16), alignment=16)
        (pv + item * 8).store((pav + item * 8).load(count=8, alignment=16).to(BF16), alignment=16)


frost_qsa_bprop_finish.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def finish_host(aq: cute.Tensor, ak: cute.Tensor, av: cute.Tensor, dq: cute.Tensor, dk: cute.Tensor, dv: cute.Tensor, blocks: I32, stream: cuda.CUstream):
    frost_qsa_bprop_finish(aq, ak, av, dq, dk, dv).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)


def fake(dtype, shape, align=16):
    return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=align)


def compile_helpers(hq, hkv, arch):
    key = template_key(globals(), {"hq": hq, "hkv": hkv, "arch": arch}, "compile_helpers")
    t = cute.sym_int64()
    q, kv = fake(BF16, (t, hq, D)), fake(BF16, (t, hkv, D))
    aq, ak = fake(F32, (t, hq, D)), fake(F32, (t, hkv, D))
    delta = fake(F32, (t, hq))
    stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    opts = f"--enable-tvm-ffi --gpu-arch=sm_{arch}a"
    return {
        "prep": compile_cached(prepare_host, aq, q, delta, aq, ak, ak, I32(1), stream, options=opts, cache_key=key, symbol="qsa_backward_prepare"),
        "finish": compile_cached(finish_host, aq, ak, ak, q, kv, kv, I32(1), stream, options=opts, cache_key=key, symbol="qsa_backward_finish"),
    }


def launch_backward(compiled, grad_out, q, k, v, out_fp32, lse, ids, counts, lo, hi, first, last, dq, dk, dv, scratch, scale, stream, num_sms):
    """Enqueue three kernels with caller-owned outputs and workspace."""
    blocks = max(1, min(num_sms * 8, (q.numel() // 8 + 255) // 256))
    compiled["prep"](out_fp32, grad_out, scratch["delta"], scratch["dq_acc"], scratch["dk_acc"], scratch["dv_acc"], blocks, stream)
    compiled["main"](
        q,
        k,
        v,
        grad_out,
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
        scale,
        min(q.shape[0], num_sms),
        stream,
    )
    compiled["finish"](scratch["dq_acc"], scratch["dk_acc"], scratch["dv_acc"], dq, dk, dv, blocks, stream)
