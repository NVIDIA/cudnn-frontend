# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native-layout DSv4.1 vision RoPE VJP and packed QKV gradient.

The math follows DeepSeek-V4.1-Flash inference/vision.py at dba1be0a40aa45a94ad051997016db3960a90277.
The GPU addressing, shared-memory transpose, and fused packing are authored here.
"""

from cudnn.frost.buffers import cutedsl_requirement_error, cutedsl_state, cutedsl_too_old

if cutedsl_too_old(cutedsl_state()[1]):
    raise RuntimeError(cutedsl_requirement_error("FROST vision RoPE backward"))

import cutlass
import cutlass.cute as cute
from cutlass import utils
from cutlass._mlir.dialects import llvm


@cutlass.dsl_user_op
def _plus(a, c, b, s, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [x.ir_value(loc=loc, ip=ip) for x in (a, c, b, s)],
            "{ .reg .f32 p,q; mul.rn.f32 p,$1,$2; mul.rn.f32 q,$3,$4; add.rn.f32 $0,p,q; }",
            "=f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cutlass.dsl_user_op
def _minus(a, c, b, s, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [x.ir_value(loc=loc, ip=ip) for x in (a, c, b, s)],
            "{ .reg .f32 p,q; mul.rn.f32 p,$1,$2; mul.rn.f32 q,$3,$4; sub.rn.f32 $0,p,q; }",
            "=f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.kernel
def _frost_vision_rope_backward_hdt(
    grad_q: cute.Tensor,
    grad_k: cute.Tensor,
    grad_v: cute.Tensor,
    cosine: cute.Tensor,
    sine: cute.Tensor,
    grad_qkv: cute.Tensor,
    tokens: cutlass.Int64,
):
    thread, _, _ = cute.arch.thread_idx()
    tile, head, _ = cute.arch.block_idx()
    allocator = utils.SmemAllocator()
    shared = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((32, 66), stride=(66, 1)))
    input_t = thread % 32
    input_c = (thread // 32) * 8
    global_t = cutlass.Int64(tile) * 32 + input_t
    if global_t < tokens:
        for i in cutlass.range_constexpr(8):
            source = (cutlass.Int64(head) * 64 + input_c + i) * tokens + global_t
            scalar = cute.make_tensor(grad_k.iterator + source, cute.make_layout(1))
            shared[input_t, input_c + i] = scalar[0]
    cute.arch.sync_threads()

    output_t = thread // 8
    channel = cute.assume((thread % 8) * 8, divby=8)
    global_t = cutlass.Int64(tile) * 32 + output_t
    if global_t < tokens:
        source = (cutlass.Int64(head) * tokens + global_t) * 64 + channel
        opposite = source + 32 - 64 * (channel // 32)
        table = global_t * 32 + channel % 32
        cosines = cute.make_rmem_tensor(8, cutlass.Float32)
        sines = cute.make_rmem_tensor(8, cutlass.Float32)
        cute.autovec_copy(cute.make_tensor(cosine.iterator + table, cute.make_layout(8)), cosines)
        cute.autovec_copy(cute.make_tensor(sine.iterator + table, cute.make_layout(8)), sines)
        for group in cutlass.range_constexpr(2):
            values = cute.make_rmem_tensor(8, cutlass.BFloat16)
            others = cute.make_rmem_tensor(8, cutlass.BFloat16)
            if cutlass.const_expr(group == 0):
                cute.autovec_copy(cute.make_tensor(grad_q.iterator + source, cute.make_layout(8)), values)
                cute.autovec_copy(cute.make_tensor(grad_q.iterator + opposite, cute.make_layout(8)), others)
            else:
                at = output_t * 66 + channel
                pair = at + 32 - 64 * (channel // 32)
                cute.autovec_copy(cute.make_tensor(shared.iterator + at, cute.make_layout(8)), values)
                cute.autovec_copy(cute.make_tensor(shared.iterator + pair, cute.make_layout(8)), others)
            if channel < 32:
                for i in cutlass.range_constexpr(8):
                    values[i] = _plus(values[i].to(cutlass.Float32), cosines[i], others[i].to(cutlass.Float32), sines[i]).to(cutlass.BFloat16)
            else:
                for i in cutlass.range_constexpr(8):
                    values[i] = _minus(values[i].to(cutlass.Float32), cosines[i], others[i].to(cutlass.Float32), sines[i]).to(cutlass.BFloat16)
            destination = ((global_t * 3 + group) * 16 + head) * 64 + channel
            cute.autovec_copy(values, cute.make_tensor(grad_qkv.iterator + destination, cute.make_layout(8)))
        values_v = cute.make_rmem_tensor(8, cutlass.BFloat16)
        cute.autovec_copy(cute.make_tensor(grad_v.iterator + source, cute.make_layout(8)), values_v)
        destination_v = ((global_t * 3 + 2) * 16 + head) * 64 + channel
        cute.autovec_copy(values_v, cute.make_tensor(grad_qkv.iterator + destination_v, cute.make_layout(8)))


_frost_vision_rope_backward_hdt.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def launch(
    grad_q: cute.Tensor,
    grad_k: cute.Tensor,
    grad_v: cute.Tensor,
    cosine: cute.Tensor,
    sine: cute.Tensor,
    grad_qkv: cute.Tensor,
    stream,
    tokens: cutlass.Int64,
):
    _frost_vision_rope_backward_hdt(grad_q, grad_k, grad_v, cosine, sine, grad_qkv, tokens).launch(
        grid=((tokens + 31) // 32, 16, 1), block=(256, 1, 1), stream=stream
    )
