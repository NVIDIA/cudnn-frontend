# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One compiled launcher with one fresh, aligned scratch allocation."""

from functools import lru_cache
import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cuda.bindings import driver as cuda
import torch

from cudnn.frost.compiled_cache import compile_cached, template_key
from . import _metadata
from .kernels import fprop_dense, fprop_prepare, fprop_sm100

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SOURCE_DIGESTS = tuple(module.FROST_SOURCE_DIGEST for module in (_metadata, fprop_dense, fprop_prepare, fprop_sm100))
_N64_SOURCE_DIGESTS = tuple(
    hashlib.sha256(Path(__file__).with_name("kernels").joinpath(name).read_bytes()).hexdigest() for name in ("fprop_n64_sm107.py", "_fprop_n64_ops.py")
)


@cute.jit
def _launch(
    q,
    k,
    v,
    sel,
    lo,
    hi,
    bases,
    origins,
    first,
    last,
    out,
    out32,
    lse,
    scratch,
    scale: Float32,
    stream: cuda.CUstream,
    arch: cutlass.Constexpr,
    gated: cutlass.Constexpr,
    n64: cutlass.Constexpr,
):
    total, blocks = Int64(q.shape[0]), Int64(sel.shape[1])
    ids_offset = Int64(0)
    if cutlass.const_expr(gated):
        tiles = cute.ceil_div(total, 128)
        words = cute.ceil_div(total, 256) * 8
        pieces = cute.ceil_div(Int64(cute.size(k)), 1024) * 4
        modes_offset = pieces
        ranges_offset = modes_offset + cute.ceil_div(tiles, 4) * 4
        bits_offset = ranges_offset + cute.ceil_div(tiles * 2, 4) * 4
        order_offset = bits_offset + total * words
        ids_offset = order_offset + cute.ceil_div(tiles, 4) * 4
        partials = cute.make_tensor(scratch.iterator, cute.make_layout((pieces,)))
        modes = cute.make_tensor(scratch.iterator + modes_offset, cute.make_layout((tiles,)))
        ranges = cute.make_tensor(scratch.iterator + ranges_offset, cute.make_layout((tiles, 2), stride=(2, 1)))
        bits = cute.make_tensor(scratch.iterator + bits_offset, cute.make_layout((total, words), stride=(words, 1)))
        order = cute.make_tensor(scratch.iterator + order_offset, cute.make_layout((tiles,)))
        ids = cute.make_tensor(scratch.iterator + ids_offset, cute.make_layout((total, blocks), stride=(blocks, 1)))
        counts_offset = ids_offset + cute.ceil_div(total * blocks, 4) * 4
        counts = cute.make_tensor(scratch.iterator + counts_offset, cute.make_layout((total,)))
        flat_k = cute.make_tensor(k.iterator, cute.make_layout((cute.size(k),)))
        flat_v = cute.make_tensor(v.iterator, cute.make_layout((cute.size(v),)))
        fprop_prepare.prepare(flat_k, flat_v, sel, lo, hi, bases, origins, first, last, partials, modes, ranges, bits, order, ids, counts, stream, arch == 107)
        total, hq, dim = q.shape
        hkv = k.shape[1]
        q_view = cute.make_tensor(q.iterator, cute.make_layout((total, dim, hq), stride=(hq * dim, 1, dim)))
        k_view = cute.make_tensor(k.iterator, cute.make_layout((total, dim, hkv), stride=(hkv * dim, 1, dim)))
        v_view = cute.make_tensor(v.iterator, cute.make_layout((dim, total, hkv), stride=(1, hkv * dim, dim)))
        flat_bits = cute.make_tensor(bits.iterator, cute.make_layout((cute.size(bits),)))
        flat_out = cute.make_tensor(out.iterator, cute.make_layout((cute.size(out),)))
        flat_out32 = cute.make_tensor(out32.iterator, cute.make_layout((cute.size(out32),)))
        flat_lse = cute.make_tensor(lse.iterator, cute.make_layout((cute.size(lse),)))
        fprop_dense._attn_launch(
            q_view, k_view, v_view, flat_bits, ranges, modes, order, flat_out, flat_out32, flat_lse, scale, Int32(bits.shape[1]), stream, hq // hkv, hq
        )
        fprop_sm100._fprop_host(q, k, v, out, out32, lse, ids, counts, lo, hi, first, last, modes, scale, stream, arch)
    else:
        ids = cute.make_tensor(scratch.iterator, cute.make_layout((total, blocks), stride=(blocks, 1)))
        counts_offset = cute.ceil_div(total * blocks, 4) * 4
        counts = cute.make_tensor(scratch.iterator + counts_offset, cute.make_layout((total,)))
        _metadata._compact_host(sel, lo, hi, first, last, ids, counts, stream, arch == 107)
        if cutlass.const_expr(n64):
            from .kernels import fprop_n64_sm107

            fprop_n64_sm107._fprop_host(q, k, v, out, out32, lse, ids, counts, lo, hi, first, last, None, scale, stream, arch)
        else:
            fprop_sm100._fprop_host(q, k, v, out, out32, lse, ids, counts, lo, hi, first, last, None, scale, stream, arch)


@lru_cache(maxsize=None)
def _compile_args(hq, hkv, gated):
    total, blocks, scratch_elements = (cute.sym_int64() for _ in range(3))

    def tensor(dtype, shape, alignment=4):
        return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=alignment)

    q = tensor(cutlass.BFloat16, (total, hq, 256), 16)
    kv = tensor(cutlass.BFloat16, (total, hkv, 256), 16)
    rows, spans = tensor(Int32, (total,)), tensor(Int32, (blocks,))
    return (
        q,
        kv,
        kv,
        tensor(cutlass.Uint8, (total, blocks), 1),
        rows,
        rows,
        rows,
        rows,
        spans,
        spans,
        q,
        tensor(Float32, (total, hq, 256), 16),
        tensor(Float32, (hq, total)),
        tensor(Int32, (scratch_elements,), 16),
        Float32(1),
        make_fake_stream(use_tvm_ffi_env_stream=False),
    )


@lru_cache(maxsize=None)
def _compile_key(hq, hkv, arch, gated, n64=False):
    return template_key(
        globals(),
        {
            **locals(),
            "sources": SOURCE_DIGESTS,
            "sm107": fprop_sm100._FPROP_SM107_SOURCE_DIGEST if arch == 107 else None,
            "n64_sources": _N64_SOURCE_DIGESTS if n64 else None,
        },
        "compile_kernel",
    )


def compile_kernel(hq, hkv, arch, gated, n64=False):
    key = _compile_key(hq, hkv, arch, gated, n64)
    return compile_cached(
        _launch, *_compile_args(hq, hkv, gated), arch, gated, n64, cache_key=key, symbol="qsa_forward_host", options=f"--enable-tvm-ffi --gpu-arch=sm_{arch}a"
    )


def launch(impl, arch, q, k, v, sel_blocks, lo, hi, blk_base, tok_base, block_first, block_last, out, out_fp32, lse, scale, stream):
    total, hq, _ = q.shape
    gated = 512 < total <= 16384
    # The 64-key Rubin forward is specialised for 12 heads per KV head; 6:1 stays on the shared gathered body.
    n64 = arch == 107 and hq == 12 * k.shape[1] and total >= 32768
    compiled = compile_kernel(hq, k.shape[1], arch, gated, n64)
    align4 = lambda value: (value + 3) // 4 * 4
    scratch_elements = align4(total * sel_blocks.shape[1]) + total
    if gated:
        tiles, words = (total + 127) // 128, ((total + 255) // 256) * 8
        pieces = ((k.numel() + 1023) // 1024) * 4
        scratch_elements += pieces + 2 * align4(tiles) + align4(2 * tiles) + total * words
    scratch = torch.empty(scratch_elements, dtype=torch.int32, device=q.device)
    compiled(q, k, v, sel_blocks, lo, hi, blk_base, tok_base, block_first, block_last, out, out_fp32, lse, scratch, Float32(scale), stream)
